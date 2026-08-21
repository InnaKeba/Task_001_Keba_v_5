import os
import sqlite3
import operator
import json
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# Імпортуємо наші інструменти та базу знань
from tools import run_query, describe_stats, detect_anomalies, linear_forecast, delete_records
from knowledge import search_knowledge

load_dotenv()

# 1. Схеми вхідних/вихідних даних для LLM (Pydantic)
class Plan(BaseModel):
    goal: str = Field(description='Головна ціль аналітичної задачі')
    steps: list[str] = Field(description='Список логічних кроків для досягнення цілі')

class ReplanDecision(BaseModel):
    action: Literal['continue', 'replan', 'finish'] = Field(
        description='continue=наступний крок, replan=змінити план, finish=завершити'
    )
    updated_steps: list[str] | None = Field(default=None, description='Нові кроки (якщо replan)')
    reasoning: str = Field(description='Пояснення прийнятого рішення')

# 2. Схема стану агента
class PlanExecuteState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: list[str]
    current_step: int
    results: list[str]
    completed: bool
    step_count: int

# 3. Налаштування LLM та інструментів
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    timeout=120 
)

planner_llm = llm.with_structured_output(Plan)
replanner_llm = llm.with_structured_output(ReplanDecision)

tools = [run_query, describe_stats, detect_anomalies, linear_forecast, search_knowledge, delete_records]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

RISKY_TOOLS = {'delete_records'}
MAX_STEPS = 10 

# 4. Вузли графа
def planner_node(state: PlanExecuteState) -> dict:
    """Генерує початковий покроковий план."""
    user_msg = state['messages'][0].content if state['messages'] else ''
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    
    prompt = (
        f'Ти Data Analyst. Створи план для задачі: {user_msg}\n\n'
        f'Доступні інструменти:\n{tool_descriptions}\n\n'
        f'Розбий задачу на логічні кроки.'
    )
    plan = planner_llm.invoke(prompt)
    return {
        'plan': plan.steps,
        'current_step': 0,
        'results': [],
        'messages': [AIMessage(content=f'План:\n- ' + '\n- '.join(plan.steps))],
        'completed': False,
        'step_count': 0
    }

def executor_node(state: PlanExecuteState) -> dict:
    """Виконує один крок плану (міні-ReAct з викликом інструментів та HITL)."""
    step_idx = state['current_step']
    plan = state['plan']
    step_count = state.get('step_count', 0)

    if step_idx >= len(plan):
        return {'completed': True}

    current_step = plan[step_idx]
    response = llm_with_tools.invoke(
        f'Виконай крок: "{current_step}"\nПопередні результати аналізу: {state.get("results", [])}'
    )

    result_text = response.content
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_outputs = []
        for tc in response.tool_calls:
            # Ризиковані інструменти вимагають підтвердження від аналітика
            if tc['name'] in RISKY_TOOLS:
                approval = interrupt({
                    'action': tc['name'],
                    'args': tc['args'],
                    'message': f"Важливо! Спроба видалення даних: {tc['name']} з параметрами {tc['args']}."
                })
                
                if isinstance(approval, dict) and approval.get('approved'):
                    tool_fn = tools_by_name.get(tc['name'])
                    res = tool_fn.invoke(tc['args'])
                    tool_outputs.append(f"{tc['name']}: {res} (ПІДТВЕРДЖЕНО)")
                else:
                    reason = approval.get('reason', 'Відхилено аналітиком') if isinstance(approval, dict) else 'Відхилено'
                    tool_outputs.append(f"{tc['name']}: СКАСОВАНО ({reason})")
            
            else:
                tool_fn = tools_by_name.get(tc['name'])
                if tool_fn:
                    tool_outputs.append(f'{tc["name"]}: {tool_fn.invoke(tc["args"])}')
                    
        result_text = " | ".join(tool_outputs)

    if not result_text: result_text = "Виконано (без текстового виводу)."
    step_result_msg = f'Крок {step_idx + 1} виконано: {result_text}'

    return {
        'current_step': step_idx + 1,
        'results': [*state.get('results', []), step_result_msg],
        'messages': [AIMessage(content=step_result_msg)],
        'step_count': step_count + 1
    }

def replanner_node(state: PlanExecuteState) -> dict:
    """Оцінює прогрес і вирішує, що робити далі."""
    plan = state['plan']
    step_idx = state['current_step']
    results = state.get('results', [])
    step_count = state.get('step_count', 0)

    # Захисний механізм: детекція зациклень та ліміт кроків
    if step_count >= MAX_STEPS:
        return {'completed': True, 'messages': [AIMessage(content='Досягнуто ліміт кроків (MAX_STEPS). Завершую роботу.')]}

    if step_idx >= len(plan):
        return {'completed': True, 'messages': [AIMessage(content='Аналіз завершено.')]}

    prompt = (
        f'План: {plan}\nВиконано кроків: {step_idx}/{len(plan)}\n'
        f'Результати: {results}\nЗалишилось: {plan[step_idx:]}\n'
        f'Виріши: continue (продовжити), replan (змінити план) або finish (якщо все готово).'
    )
    decision = replanner_llm.invoke(prompt)

    if decision.action == 'finish':
        return {'completed': True, 'messages': [AIMessage(content=f'Завершено: {decision.reasoning}')]}
    elif decision.action == 'replan' and decision.updated_steps:
        return {'plan': decision.updated_steps, 'current_step': 0, 'messages': [AIMessage(content=f'План змінено: {decision.reasoning}')]}
    return {}

def should_end(state: PlanExecuteState) -> Literal['executor', '__end__']:
    if state.get('completed'): return '__end__'
    return 'executor'

# 5. Побудова графа та пам'яті
graph = StateGraph(PlanExecuteState)
graph.add_node('planner', planner_node)
graph.add_node('executor', executor_node)
graph.add_node('replanner', replanner_node)

graph.add_edge(START, 'planner')
graph.add_edge('planner', 'executor')
graph.add_edge('executor', 'replanner')
graph.add_conditional_edges('replanner', should_end)

conn = sqlite3.connect('agent_state.db', check_same_thread=False)
saver = SqliteSaver(conn)
app = graph.compile(checkpointer=saver)

# 6. Основний сценарій виконання
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # СЦЕНАРІЙ 1: повний аналітичний процес з прогнозуванням та детекцією аномалій
    print("="*60)
    print("СЦЕНАРІЙ 1: АНАЛІЗ ПРОДАЖІВ 2026")
    print("="*60)
    
    query = "Проаналізуй дані продажів за 2026 рік: обчисли щомісячну виручку, вияви аномалії та побудуй прогноз на наступний квартал (3 періоди)."
    config1 = {'configurable': {'thread_id': 'analytics_session_003'}}
    
    app.invoke(
        {'messages': [HumanMessage(content=query)], 'plan': [], 'current_step': 0, 'results': [], 'completed': False, 'step_count': 0},
        config=config1
    )
    
    state1 = app.get_state(config1)
    print("\n[ЛОГ ТРАЄКТОРІЇ]")
    for step in state1.values.get('results', []):
        print(step)

    # СЦЕНАРІЙ 2: HITL
    print("\n\n" + "="*60)
    print("СЦЕНАРІЙ 2: HITL (ПІДТВЕРДЖЕННЯ ТА ВІДХИЛЕННЯ)")
    print("="*60)
    
    query_hitl = "Видали всі записи з таблиці sales_2026 де продаж менше 1000."
    config2 = {'configurable': {'thread_id': 'hitl_session_004'}}
    
    print("[Система] Запуск агента...")
    for event in app.stream(
        {'messages': [HumanMessage(content=query_hitl)], 'plan': [], 'current_step': 0, 'results': [], 'completed': False, 'step_count': 0}, 
        config2
    ):
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"][0].value
            print(f"\nГРАФ ПРИЗУПИНЕНО!")
            print(f"Повідомлення: {interrupt_data['message']}")
            break
            
    print("\n[Аналітик] Натискає 'ВІДХИЛИТИ' (захист від втрати даних)...")
    app.invoke(Command(resume={"approved": False, "reason": "Заборонено видаляти дані фінансового року"}), config=config2)
    
    state2 = app.get_state(config2)
    print("\n[ЛОГ ТРАЄКТОРІЇ ПІСЛЯ ВІДМОВИ]")
    print(state2.values.get('results')[-1])