import json
import chromadb
from langchain_core.tools import tool

# Ініціалізація векторної бази знань для Data Analyst
chroma_client = chromadb.PersistentClient(path="./chroma_db")
knowledge_base = chroma_client.get_or_create_collection(name="data_analyst_knowledge")

# База знань містить теоретичні матеріали для Data Analyst, включаючи:
documents = [
    "Описова статистика: Середнє значення (mean) показує загальний центр даних, а медіана (median) — значення, що ділить вибірку навпіл і є стійким до викидів.",
    "Описова статистика: Стандартне відхилення (std) вимірює розкид даних навколо середнього значення. Чим воно вище, тим більша варіативність даних.",
    "Методи детекції аномалій: Метод Z-score (стандартна оцінка) ідентифікує аномалії як значення, що відхиляються від середнього більше ніж на 2 або 3 стандартні відхилення.",
    "Методи детекції аномалій: Метод IQR (міжквартильний розмах) визначає викиди як значення, що лежать нижче Q1 - 1.5*IQR або вище Q3 + 1.5*IQR. Він краще працює з ненормальними розподілами.",
    "Типи візуалізацій: Лінійний графік (line chart) найкраще підходить для відображення трендів у часі (наприклад, продажі по місяцях). Стовпчаста діаграма (bar chart) ідеальна для порівняння категорій.",
    "Типи візуалізацій: Діаграма розсіювання (scatter plot) використовується для пошуку кореляції між двома числовими змінними. Box plot показує розподіл та викиди.",
    "Основи прогнозування: Лінійне прогнозування (linear forecast) використовує історичні дані для побудови прямої лінії тренду та екстраполяції майбутніх значень. Підходить для стабільних трендів.",
    "SQL-синтаксис: Для фільтрації даних використовується оператор WHERE. Наприклад, 'WHERE year = 2026' дозволяє відібрати записи лише за 2026 рік.",
    "SQL-синтаксис: Для агрегації даних використовуються функції SUM (сума), AVG (середнє), COUNT (кількість), MIN (мінімум), MAX (максимум)."
]

doc_ids = [f"doc_{i}" for i in range(len(documents))]

knowledge_base.upsert(documents=documents, ids=doc_ids)
print(f"База знань Data Analyst успішно ініціалізована. Завантажено {knowledge_base.count()} документів")

# RAG tool
@tool
def search_knowledge(query: str) -> str:
    """Пошук інформації у базі знань для Data Analyst.
    
    Використовуй цей інструмент, коли потрібна теоретична довідка про:
    методи статистичного аналізу, типи візуалізацій, детекцію аномалій, 
    прогнозування або SQL-синтаксис. 
    НЕ використовуй для обчислень або запитів до БД.
    
    Args:
        query: Пошуковий запит (наприклад, 'як працює Z-score').
    """
    try:
        results = knowledge_base.query(query_texts=[query], n_results=2)
        
        if not results['documents'] or not results['documents'][0]:
            return json.dumps({"status": "error", "error": "Інформацію не знайдено."}, ensure_ascii=False)
            
        docs = results['documents'][0]
        found_text = "\n---\n".join(docs)
        
        return json.dumps({"status": "success", "data": {"knowledge": found_text}}, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

if __name__ == "__main__":
    # Локальний тест RAG
    test_result = search_knowledge.invoke("Який графік краще для трендів у часі?")
    print(f"\nТест RAG:\n{test_result}")