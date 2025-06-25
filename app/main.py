
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os
import re
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка пайплайна и конфигурации
try:
    pipeline = joblib.load('app/pipeline_best_model.joblib')
    logger.info("Pipeline loaded successfully")
except Exception as e:
    logger.error(f"Pipeline loading error: {str(e)}")
    raise RuntimeError(f"Ошибка загрузки пайплайна: {str(e)}")

# Конфигурация предобработки
PREPROCESSING_CONFIG = {
            "drop_columns": ["income", "unnamed_0"], # Столбец для удаления
            "round_columns": ["sleep_hours_per_day"], # Столбец для округления значений
            "index_column": "id", # Столбeц для перевода в индекс
            "type_conversions": {   # Конверсия типов данных
                            "diabetes": "int",
                            "family_history": "int",
                            "smoking": "int",
                            "obesity": "int",
                            "alcohol_consumption": "int",
                            "diet": "int",
                            "previous_heart_problems": "int",
                            "medication_use": "int",
                            "stress_level": "int",
                            "physical_activity_days_per_week": "int"
                            #"sleep_hours_per_day": "string"
            },
            "drop_na": True,  # Удалять строки с пропущенными значениями
            "drop_na_columns": "all",  # "all" или список конкретных столбцов
            "classification_threshold": 0.4908248526120897  # Порог для бинарной классификации
}


def clean_column_names(columns: List[str]) -> List[str]:
    """Нормализация имен столбцов с заменой дефисов и других символов"""
    cleaned = []
    for col in columns:
        # Замена дефисов, скобок, пробелов и других специальных символов
        col = re.sub(r'[\-\s\(\)\{\}\[\],;:!?@#\$%\^&\*\+=\|<>/`~]', '_', col.strip().lower())
        # Удаление всех не-алфавитно-цифровых символов, кроме подчеркиваний
        col = re.sub(r'[^\w_]', '', col)
        # Удаление последовательных подчеркиваний
        col = re.sub(r'_+', '_', col).strip('_')
        cleaned.append(col)
    return cleaned

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расширенная предобработка данных с удалением пропусков
    """
    try:
        logger.info("Starting data preprocessing")

        # 1. Нормализация имен столбцов (замена дефисов, скобок и спецсимволов)
        df.columns = clean_column_names(df.columns)
        logger.info(f"Cleaned columns: {df.columns.tolist()}")

        #2. Удаление указанных столбцов
        for col in PREPROCESSING_CONFIG["drop_columns"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        logger.info(f"Columns after dropping: {df.columns.tolist()}")

        # 3. Удаление строк с пропущенными значениями
        na_removed = 0
        if PREPROCESSING_CONFIG.get("drop_na", True):
            drop_na_columns = PREPROCESSING_CONFIG.get("drop_na_columns", "all")
            initial_count = len(df)

            if drop_na_columns == "all":
                df = df.dropna()
            elif isinstance(drop_na_columns, list):
                # Фильтруем только существующие столбцы
                valid_columns = [col for col in drop_na_columns if col in df.columns]
                df = df.dropna(subset=valid_columns)

            na_removed = initial_count - len(df)
            logger.info(f"Removed {na_removed} rows with missing values")

        PREPROCESSING_CONFIG["na_removed"] = na_removed
        logger.info(f"Data shape after preprocessing: {df.shape}")

        # 4. Округление значений столбцов с последующей конверсией типа данных в str
        for col in PREPROCESSING_CONFIG["round_columns"]:
            if col in df.columns:
                # Обработка отсутствующих значений
                s = pd.to_numeric(df[col], errors='coerce')
                df[col] = s.round(5).astype(str)
        logger.info("Rounding and string conversion completed")

        # 5. Конверсия типов данных
        type_map = {
            "int": lambda x: x.astype(int)
            #"int": lambda x: pd.to_numeric(x, errors='coerce').astype(int)
            #"string": lambda x: x.astype(str)
        }

        for col, col_type in PREPROCESSING_CONFIG["type_conversions"].items():
            if col in df.columns:
                df[col] = type_map[col_type](df[col])
        logger.info("Type conversions completed")

        # 6. Перевод столбца в индексы
        if PREPROCESSING_CONFIG.get("index_column") and PREPROCESSING_CONFIG["index_column"] in df.columns:
            df = df.set_index(PREPROCESSING_CONFIG["index_column"])
        logger.info(f"Set index: {PREPROCESSING_CONFIG['index_column']}")



        return df

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
        raise

def apply_threshold(predictions_proba: np.ndarray) -> np.ndarray:
    """Применение порога к вероятностям для получения бинарных предсказаний"""
    try:
        threshold = float(PREPROCESSING_CONFIG.get("classification_threshold", 0.5))
        # Гарантируем, что это массив чисел с плавающей точкой
        predictions_proba = predictions_proba.astype(float)
        return (predictions_proba >= threshold).astype(int)
    except Exception as e:
        logger.error(f"Threshold application error: {str(e)}", exc_info=True)
        raise ValueError(f"Threshold application error: {str(e)}") from e


# Модель для API запроса
class PredictionRequest(BaseModel):
    file_path: str
    return_probabilities: Optional[bool] = False


# Веб-интерфейс
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": PREPROCESSING_CONFIG
    })


# Обработка формы
@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request, file_path: str = Form(...),
                       return_probabilities: bool = Form(False)):
    try:
        # Сброс счетчика удаленных строк
        if "na_removed" in PREPROCESSING_CONFIG:
            PREPROCESSING_CONFIG.pop("na_removed", None)

        # Проверка существования файла
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": f"Файл не найден: {file_path}"
            })

        # Чтение CSV
        try:
            df = pd.read_csv(file_path)
            logger.info(f"CSV loaded successfully: {file_path}")
        except Exception as e:
            logger.error(f"CSV reading error: {str(e)}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": f"Ошибка чтения CSV: {str(e)}"
            })


        # Проверка данных
        if df.empty:
            logger.warning("Empty CSV file")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": "CSV файл пуст"
            })

        initial_row_count = len(df)

        # Предобработка данных
        try:
            original_columns = df.columns.tolist()
            df = preprocess_data(df)
            processed_columns = df.columns.tolist()
            na_removed = PREPROCESSING_CONFIG.get("na_removed", 0)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": f"Ошибка предобработки: {str(e)}"
            })

        # Проверка после предобработки
        if df.empty:
            logger.warning("No data left after preprocessing")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": "После предобработки данных не осталось"
            })

        # Выполнение предсказания
        try:
            logger.info("Starting prediction")
            # Проверяем, поддерживает ли модель предсказание вероятностей
            if return_probabilities and hasattr(pipeline, "predict_proba"):
                predictions_proba = pipeline.predict_proba(df)

                # Для бинарной классификации берем вероятности положительного класса
                if predictions_proba.shape[1] > 1:
                    predictions_proba = predictions_proba[:, 1]
                else:
                    predictions_proba = predictions_proba[:, 0]

                # Гарантируем числовой формат
                predictions_proba = predictions_proba.astype(float)
                predictions = apply_threshold(predictions_proba)
                predictions_data = predictions_proba
            else:
                predictions = pipeline.predict(df)
                # Гарантируем числовой формат для последующих операций
                predictions = predictions.astype(float)
                predictions_data = predictions

            logger.info(f"Prediction completed: {len(predictions)} results")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": f"Ошибка предсказания: {str(e)}"
            })

        # Преобразование предсказаний для вывода
        try:
            predictions_list = predictions.tolist()
            predictions_data_list = predictions_data.tolist() if hasattr(predictions_data, 'tolist') else predictions_data

            # Для бинарной классификации подсчитываем классы
            positive_count = sum(1 for p in predictions_list if p >= 0.5)
            negative_count = len(predictions_list) - positive_count
        except Exception as e:
            logger.error(f"Result conversion error: {str(e)}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "config": PREPROCESSING_CONFIG,
                "error": f"Ошибка обработки результатов: {str(e)}"
            })

        # Форматирование результата
        return templates.TemplateResponse("index.html", {
            "request": request,
            "config": PREPROCESSING_CONFIG,
            "file_path": file_path,
            "original_columns": original_columns,
            "processed_columns": processed_columns,
            "predictions": predictions_list,
            "predictions_data": predictions_data_list,
            "count": len(predictions_list),
            "initial_row_count": initial_row_count,
            "na_removed": na_removed,
            "return_probabilities": return_probabilities,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sample_data": df.head(5).to_dict(orient="records")
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "config": PREPROCESSING_CONFIG,
            "error": f"Непредвиденная ошибка: {str(e)}"
        })


# API эндпоинт
@app.post("/predict")
async def predict_api(request: PredictionRequest):
    file_path = request.file_path
    return_probabilities = request.return_probabilities

    # Сброс счетчика удаленных строк
    if "na_removed" in PREPROCESSING_CONFIG:
        PREPROCESSING_CONFIG.pop("na_removed", None)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Файл не найден")

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV файл пуст")

        initial_row_count = len(df)

        # Предобработка данных
        original_columns = df.columns.tolist()
        df = preprocess_data(df)
        processed_columns = df.columns.tolist()
        na_removed = PREPROCESSING_CONFIG.get("na_removed", 0)

        if df.empty:
            raise HTTPException(status_code=400, detail="После предобработки данных не осталось")

        # Выполнение предсказания
        if return_probabilities and hasattr(pipeline, "predict_proba"):
            predictions_proba = pipeline.predict_proba(df)

            if predictions_proba.shape[1] > 1:
                predictions_proba = predictions_proba[:, 1]
            else:
                predictions_proba = predictions_proba[:, 0]

            predictions_proba = predictions_proba.astype(float)
            predictions = apply_threshold(predictions_proba)
            predictions_data = predictions_proba
        else:
            predictions = pipeline.predict(df)
            predictions = predictions.astype(float)
            predictions_data = predictions

        # Преобразование результатов
        predictions_list = predictions.tolist()
        predictions_data_list = predictions_data.tolist() if hasattr(predictions_data, 'tolist') else predictions_data

        return {
            "file": file_path,
            "original_columns": original_columns,
            "processed_columns": processed_columns,
            "predictions": predictions_list,
            "predictions_data": predictions_data_list,
            "return_probabilities": return_probabilities,
            "count": len(predictions_list),
            "initial_row_count": initial_row_count,
            "na_removed": na_removed,
            "threshold": PREPROCESSING_CONFIG.get("classification_threshold"),
            "sample_data": df.head(2).to_dict(orient="records")
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
