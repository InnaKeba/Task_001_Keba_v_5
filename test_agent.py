import pytest
import json
from pydantic import ValidationError
from tools import (
    RunQueryInput, DescribeStatsInput, DetectAnomaliesInput, 
    DeleteRecordsInput, describe_stats, detect_anomalies
)

# 1. Тести валідації вхідних даних Pydantic
def test_run_query_input_valid():
    """Тест 1: Валідна схема для SQL-запиту з правильною агрегацією"""
    data = RunQueryInput(table="sales", columns=["amount"], aggregation="sum")
    assert data.aggregation == "SUM"

def test_run_query_input_invalid_agg():
    """Тест 2: Невалідна агрегація викликає помилку Pydantic"""
    with pytest.raises(ValidationError):
        RunQueryInput(table="sales", columns=["amount"], aggregation="UNKNOWN")

def test_describe_stats_input_invalid_json():
    """Тест 3: Передача звичайного тексту замість JSON-масиву викликає помилку"""
    with pytest.raises(ValidationError):
        DescribeStatsInput(data_json="просто текст")

def test_delete_records_input_short_where():
    """Тест 4: Занадто коротка умова WHERE блокується (захист від DROP TABLE)"""
    with pytest.raises(ValidationError):
        DeleteRecordsInput(table="users", where_clause="id")

def test_detect_anomalies_invalid_method():
    """Тест 5: Використання неіснуючого методу детекції аномалій"""
    with pytest.raises(ValidationError):
        DetectAnomaliesInput(values=[1, 2, 3, 4], method="magic")


# 2. Тести виконання інструментів

def test_describe_stats_execution():
    """Тест 6: Перевірка математичних розрахунків інструменту статистики"""
    result_str = describe_stats.invoke({"data_json": "[10, 20, 30]"})
    result = json.loads(result_str)
    
    assert result["status"] == "success"
    assert result["data"]["mean"] == 20
    assert result["data"]["max"] == 30

def test_detect_anomalies_execution():
    """Тест 7: Перевірка пошуку викидів методом Z-score"""
    normal_data = [10, 11, 12, 10, 11, 10, 12, 11, 10, 11]
    test_data = normal_data + [100]
    
    result_str = detect_anomalies.invoke({"values": test_data, "method": "z-score"})
    result = json.loads(result_str)
    
    assert result["status"] == "success"
    assert len(result["data"]["anomalies"]) > 0
    assert result["data"]["anomalies"][0]["value"] == 100