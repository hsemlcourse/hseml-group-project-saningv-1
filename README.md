# SMS Spam Detection

**Задача:** бинарная классификация SMS-сообщений: `spam` или `ham`.

**Студент:** Gleb Sanin

**Источник данных:** SMS Spam Collection, опубликованный UCI и продублированный на Kaggle:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Что внутри

В репозитории есть полный локальный пайплайн: загрузка и очистка данных, обучение модели, API на FastAPI и простой Streamlit-интерфейс для ручной проверки SMS.

После удаления точных дублей в датасете остаётся 5 158 сообщений, из них 642 относятся к spam. Датасет небольшой, это ограничение было согласовано. В модели используется не только сырой текст, но и TF-IDF признаки вместе с простыми числовыми признаками сообщения.

## Структура

```text
.
├── app
│   ├── api.py                  # FastAPI: /health и /predict
│   └── streamlit_app.py        # UI для ручного ввода SMS
├── data
│   ├── raw/                    # локальный архив, не коммитится
│   └── processed/              # обработанный CSV, не коммитится
├── models/                     # локальные .joblib модели, не коммитятся
├── report
│   ├── images/                 # графики и изображения деплоя
│   ├── cp1_results.csv
│   └── report.md
├── src
│   ├── data.py
│   ├── deployment_artifacts.py
│   ├── features.py
│   ├── model_service.py
│   ├── split.py
│   └── train_baseline.py
├── tests
└── requirements.txt
```

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data
python -m src.data
python -m src.train_baseline
python -m src.deployment_artifacts
python -m src.build_report_pdf
```

## API

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"URGENT! You won a free prize. Call now to claim cash."}'
```

Пример ответа:

```json
{
  "label": "spam",
  "target": 1,
  "spam_score": 0.9902
}
```

## Интерфейс

```bash
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

После запуска интерфейс доступен по адресу:

```text
http://127.0.0.1:8501
```

## Docker

```bash
docker build -t sms-spam-detector .
docker run --rm -p 8000:8000 sms-spam-detector
```

## Проверки

```bash
pytest -q
ruff check src app tests --line-length 120
flake8 src app tests --max-line-length=120
```
