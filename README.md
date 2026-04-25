# SMS Spam Detection

**Задача:** бинарная классификация SMS-сообщений: `spam` или `ham`.

**Источник данных:** SMS Spam Collection, опубликованный UCI и продублированный на Kaggle:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Что сделано к CP1

На CP1 я собрал воспроизводимый baseline-пайплайн для классической задачи фильтрации SMS-спама: загрузка сырого корпуса, очистка, ручные признаки поверх текста, стратифицированный `train/val/test` split и несколько первых моделей.

У датасета есть важное ограничение: в сыром виде это не большая табличная витрина, а компактный текстовый корпус. После удаления дублей остаётся 5 158 сообщений: 4 516 обычных и 642 спам-сообщения. Формально это меньше порога в 10 000 строк и исходно выглядит как две колонки, но для текстовой классификации это не означает “два признака”. Текст разворачивается в высокоразмерное пространство через `CountVectorizer`/`TfidfVectorizer`, а поверх него добавлены интерпретируемые признаки, которые хорошо описывают механику SMS-спама: длина сообщения, число слов, цифр, телефонов, валютных символов, uppercase-доля, пунктуация и CTA-слова вроде `call`, `free`, `win`, `claim`.

Проект и датасет согласованы, поэтому на CP1 я явно фиксирую это ограничение и компенсирую его двумя вещами: аккуратной валидацией без leakage и признаками, которые можно объяснить без магии модели.

## Структура

```text
.
├── data
│   ├── raw/                         # локальный архив sms_spam_collection.zip, не коммитится
│   └── processed/                   # обработанный CSV, не коммитится
├── models/                          # обученная модель, не коммитится
├── report
│   ├── images/                      # графики EDA
│   ├── cp1_data_summary.json
│   ├── cp1_results.csv
│   └── report.md
├── src
│   ├── config.py                    # пути и seed
│   ├── data.py                      # загрузка и очистка данных
│   ├── eda.py                       # графики и summary для отчёта
│   ├── features.py                  # ручные текстовые признаки
│   ├── split.py                     # стратифицированный split
│   └── train_baseline.py            # первые модели и метрики
├── tests/test.py
└── requirements.txt
```

## Воспроизведение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data
python -m src.data
python -m src.eda
python -m src.train_baseline

python -m pytest tests/test.py -q
ruff check src/ --line-length 120
```

Сырой архив лежит локально в `data/raw/sms_spam_collection.zip`, но не коммитится. Команда `python -m src.download_data` скачивает его из UCI. Ручной вариант:

```bash
curl -L -sS https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip -o data/raw/sms_spam_collection.zip
```

## Метрика

Основная метрика для CP1 - `F1` по классу `spam`. В этой задаче `accuracy` легко выглядит хорошо из-за дисбаланса классов, но для фильтра спама важен баланс между двумя ошибками: пропустить спам и ошибочно пометить нормальное сообщение как спам. Поэтому рядом также считаются `precision_spam` и `recall_spam`.

## Результаты CP1

| Модель | Split | Accuracy | Precision spam | Recall spam | F1 spam |
|---|---:|---:|---:|---:|---:|
| CountVectorizer + MultinomialNB | test | 0.979 | 0.966 | 0.866 | 0.913 |
| TF-IDF + LogisticRegression | test | 0.961 | 0.986 | 0.701 | 0.819 |
| TF-IDF + LinearSVC | val | 0.988 | 1.000 | 0.906 | 0.951 |
| TF-IDF + LinearSVC | test | 0.978 | 0.955 | 0.866 | 0.908 |
| TF-IDF + ручные признаки + LogisticRegression | test | 0.983 | 1.000 | 0.866 | 0.928 |

На validation лучшая модель - `TF-IDF + LinearSVC`. На test сильнее выглядит `TF-IDF + ручные признаки + LogisticRegression`, но для выбора модели я пока ориентируюсь на validation, чтобы не подгонять решение под test. В CP2 логично расширить сетку экспериментов, добавить кросс-валидацию и проверить, стабилен ли выигрыш ручных признаков.
