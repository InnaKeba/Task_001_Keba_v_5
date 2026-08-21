import json
import statistics
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

# 1. run_query
class RunQueryInput(BaseModel):
    table: str = Field(description="Назва таблиці баз даних (наприклад, 'sales_2026')")
    columns: list[str] = Field(description="Список колонок для вибірки")
    where_clause: str | None = Field(default=None, description="Умова фільтрації SQL (наприклад, 'month = 1')")
    aggregation: str | None = Field(default=None, description="Функція агрегації (SUM, AVG, COUNT, MIN, MAX)")

    @field_validator('aggregation')
    @classmethod
    def validate_aggregation(cls, v: str | None) -> str | None:
        if v:
            valid_aggs = ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX']
            if v.upper() not in valid_aggs:
                raise ValueError(f"Агрегація '{v}' не підтримується. Використовуйте: {valid_aggs}")
            return v.upper()
        return v

@tool(args_schema=RunQueryInput)
def run_query(table: str, columns: list[str], where_clause: str | None = None, aggregation: str | None = None) -> str:
    """Виконує SQL-подібний запит до бази даних та повертає mock-дані."""
    try:
        # Mock-логіка для імітації запиту
        mock_data = [15000, 18000, 14500, 22000, 25000] if "sales" in table.lower() else [10, 20, 30]
        result = {
            "status": "success",
            "data": {
                "query_info": f"SELECT {aggregation + '(' if aggregation else ''}{', '.join(columns)}{')' if aggregation else ''} FROM {table} WHERE {where_clause}",
                "result": sum(mock_data) if aggregation == 'SUM' else mock_data
            }
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


# 2. describe_stats
class DescribeStatsInput(BaseModel):
    data_json: str = Field(description="JSON-рядок з масивом чисел, наприклад '[10, 20.5, 30]'")

    @field_validator('data_json')
    @classmethod
    def validate_data(cls, v: str) -> str:
        try:
            data = json.loads(v)
            if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
                raise ValueError
        except:
            raise ValueError("Вхідні дані мають бути валідним JSON масивом чисел")
        return v

@tool(args_schema=DescribeStatsInput)
def describe_stats(data_json: str) -> str:
    """Обчислює описову статистику (mean, median, std, min, max) для масиву чисел."""
    try:
        data = json.loads(data_json)
        if len(data) < 2:
            raise ValueError("Потрібно мінімум 2 значення для розрахунку статистики")
        
        result = {
            "status": "success",
            "data": {
                "mean": round(statistics.mean(data), 2),
                "median": round(statistics.median(data), 2),
                "std": round(statistics.stdev(data), 2),
                "min": min(data),
                "max": max(data)
            }
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


# 3. detect_anomalies
class DetectAnomaliesInput(BaseModel):
    values: list[float] = Field(description="Список числових значень для аналізу")
    method: str = Field(description="Метод детекції: 'z-score' або 'iqr'")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v.lower() not in ['z-score', 'iqr']:
            raise ValueError("Доступні методи: 'z-score' або 'iqr'")
        return v.lower()

@tool(args_schema=DetectAnomaliesInput)
def detect_anomalies(values: list[float], method: str) -> str:
    """Виявляє аномалії у ряді даних за допомогою Z-score або IQR."""
    try:
        if len(values) < 4:
            raise ValueError("Недостатньо даних для пошуку аномалій (потрібно >= 4)")
        
        anomalies = []
        if method == 'z-score':
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            for i, val in enumerate(values):
                if std_val > 0 and abs((val - mean_val) / std_val) > 2:
                    anomalies.append({"index": i, "value": val})
        elif method == 'iqr':
            sorted_v = sorted(values)
            q1, q3 = statistics.quantiles(sorted_v, n=4)[0], statistics.quantiles(sorted_v, n=4)[2]
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for i, val in enumerate(values):
                if val < lower_bound or val > upper_bound:
                    anomalies.append({"index": i, "value": val})

        return json.dumps({"status": "success", "data": {"anomalies": anomalies, "method": method}}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


# 4. linear_forecast
class LinearForecastInput(BaseModel):
    values: list[float] = Field(description="Історичні числові дані")
    periods_ahead: int = Field(description="Кількість періодів для прогнозу вперед")

    @field_validator('periods_ahead')
    @classmethod
    def validate_periods(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Кількість періодів має бути більшою за 0")
        return v

@tool(args_schema=LinearForecastInput)
def linear_forecast(values: list[float], periods_ahead: int) -> str:
    """Будує простий лінійний прогноз на вказану кількість періодів вперед."""
    try:
        n = len(values)
        if n < 2:
            raise ValueError("Для прогнозу потрібно мінімум 2 значення")
        
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        m = numerator / denominator if denominator != 0 else 0
        c = y_mean - m * x_mean
        
        forecast = [round(m * (n + i) + c, 2) for i in range(periods_ahead)]
        
        return json.dumps({"status": "success", "data": {"forecast": forecast}}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


# 5. delete_record - ризиковий інструмент
class DeleteRecordsInput(BaseModel):
    table: str = Field(description="Назва таблиці, з якої видаляються дані")
    where_clause: str = Field(description="Умова видалення (наприклад, 'i= 5')")

    @field_validator('where_clause')
    @classmethod
    def validate_where(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError("Занадто коротка умова WHERE. Це може призвести до видалення всієї таблиці!")
        return v

@tool(args_schema=DeleteRecordsInput)
def delete_records(table: str, where_clause: str) -> str:
    """Видаляє записи з бази даних. РИЗИКОВА ДІЯ - ПОТРЕБУЄ ПІДТВЕРДЖЕННЯ."""
    try:
        return json.dumps({
            "status": "success", 
            "data": {"message": f"Успішно видалено записи з таблиці '{table}' за умовою '{where_clause}'."}
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)