from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import IMAGES_DIR, REPORT_DIR

FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]


def register_font() -> tuple[str, str]:
    regular = next((path for path in FONT_CANDIDATES if path.exists()), None)
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if regular is None:
        return "Helvetica", "Helvetica-Bold"

    pdfmetrics.registerFont(TTFont("ReportFont", str(regular)))
    if bold.exists():
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", str(bold)))
        return "ReportFont", "ReportFont-Bold"
    return "ReportFont", "ReportFont"


def make_styles() -> dict[str, ParagraphStyle]:
    regular_font, bold_font = register_font()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
        ),
        "h1": ParagraphStyle(
            "Heading",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#374151"),
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName=regular_font,
            fontSize=8.5,
            leading=11,
            backColor=colors.HexColor("#F3F4F6"),
            borderPadding=6,
            textColor=colors.HexColor("#111827"),
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style])


def bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles), leftIndent=8) for item in items],
        bulletType="bullet",
        leftIndent=16,
        bulletFontSize=6,
        spaceAfter=8,
    )


def styled_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], widths: list[float]) -> Table:
    table = Table([[p(str(cell), styles, "small") for cell in row] for row in rows], colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "ReportFont-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_image(path: Path, width: float) -> list:
    if not path.exists():
        return []
    image = Image(str(path), width=width, height=width * 0.52)
    return [image, Spacer(1, 0.35 * cm)]


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list:
    story = [
        Spacer(1, 2.6 * cm),
        p("SMS Spam Detection", styles, "title"),
        p("Классификация SMS-сообщений на spam и ham", styles, "subtitle"),
        Spacer(1, 0.6 * cm),
        p("Студент: Gleb Sanin", styles, "subtitle"),
        p("Локальный деплой: FastAPI + Streamlit", styles, "subtitle"),
        Spacer(1, 1.2 * cm),
        p(
            "В отчёте собраны данные, обработка, эксперименты, выбор финальной модели и запуск сервиса.",
            styles,
            "subtitle",
        ),
        PageBreak(),
    ]

    story += [
        p("1. Постановка задачи", styles, "h1"),
        p(
            "Я решал задачу антиспам-фильтра для коротких SMS: по тексту сообщения нужно вернуть класс "
            "<b>ham</b> или <b>spam</b>. Общая accuracy здесь не главный ориентир: если модель почти всегда "
            "говорит ham, она всё равно может выглядеть неплохо из-за дисбаланса классов.",
            styles,
        ),
        p(
            "Основная метрика - <b>F1 по классу spam</b>. Она показывает баланс между пропущенным спамом и "
            "ложной блокировкой нормальных сообщений. Дополнительно я смотрел precision, recall и accuracy.",
            styles,
        ),
        p("2. Данные", styles, "h1"),
        p(
            "Используется SMS Spam Collection: открытый корпус сообщений для задачи spam filtering. "
            "После удаления точных дублей осталось 5 158 сообщений.",
            styles,
        ),
        styled_table(
            [["Класс", "Количество"], ["ham", "4 516"], ["spam", "642"]],
            styles,
            [7 * cm, 5 * cm],
        ),
        Spacer(1, 0.35 * cm),
        p(
            "Датасет меньше формального порога в 10 000 строк, но был согласован для проекта. "
            "В сыром виде в нём две полезные колонки, зато после TF-IDF текст превращается в набор n-грамм. "
            "Дополнительно используются ручные признаки по структуре SMS.",
            styles,
        ),
        p("3. Подготовка данных", styles, "h1"),
        bullet_list(
            [
                "чтение исходного архива sms_spam_collection.zip;",
                "кодирование меток: ham = 0, spam = 1;",
                "удаление пустых сообщений и точных дублей;",
                "добавление признаков: длина, цифры, uppercase, пунктуация, URL, телефон, валюта, CTA-слова;",
                "стратифицированный split 70/15/15 на train, validation и test.",
            ],
            styles,
        ),
        p(
            "Векторизаторы находятся внутри sklearn pipeline, поэтому словарь и IDF считаются только по train. "
            "Validation и test не участвуют в построении признакового пространства.",
            styles,
        ),
    ]
    story += add_image(IMAGES_DIR / "class_balance.png", 13.5 * cm)
    story += add_image(IMAGES_DIR / "manual_feature_means.png", 13.5 * cm)

    story += [
        p("4. Baseline", styles, "h1"),
        p(
            "Первый baseline - CountVectorizer + MultinomialNB. Это быстрый и понятный вариант для текста без "
            "ручного feature engineering.",
            styles,
        ),
        styled_table(
            [["Модель", "Accuracy", "Precision spam", "Recall spam", "F1 spam"],
             ["CountVectorizer + MultinomialNB", "0.979", "0.966", "0.866", "0.913"]],
            styles,
            [6 * cm, 2.2 * cm, 2.8 * cm, 2.5 * cm, 2.2 * cm],
        ),
        Spacer(1, 0.35 * cm),
        p(
            "Baseline оказался сильным: многие spam-сообщения в корпусе шаблонные и содержат слова вроде "
            "free, call, claim, а также короткие номера и суммы.",
            styles,
        ),
        p("5. Эксперименты", styles, "h1"),
        styled_table(
            [
                ["Гипотеза", "Проверка", "Результат"],
                ["Bag-of-words даст сильный baseline", "CountVectorizer + MultinomialNB", "F1_spam = 0.913 на test"],
                ["TF-IDF будет устойчивее", "TF-IDF + LogisticRegression", "F1_spam = 0.819, recall ниже"],
                ["LinearSVC лучше разделит n-граммы", "TF-IDF + LinearSVC", "F1_spam = 0.951 на validation"],
                [
                    "Ручные признаки добавят сигнал",
                    "TF-IDF + manual features + LogisticRegression",
                    "F1_spam = 0.928 на test",
                ],
            ],
            styles,
            [5.2 * cm, 5.6 * cm, 5.0 * cm],
        ),
        Spacer(1, 0.35 * cm),
        styled_table(
            [
                ["Модель", "Split", "Accuracy", "Precision", "Recall", "F1"],
                ["CountVectorizer + MultinomialNB", "test", "0.979", "0.966", "0.866", "0.913"],
                ["TF-IDF + LogisticRegression", "test", "0.961", "0.986", "0.701", "0.819"],
                ["TF-IDF + LinearSVC", "val", "0.988", "1.000", "0.906", "0.951"],
                ["TF-IDF + LinearSVC", "test", "0.978", "0.955", "0.866", "0.908"],
                ["TF-IDF + manual + LogisticRegression", "test", "0.983", "1.000", "0.866", "0.928"],
            ],
            styles,
            [5.4 * cm, 1.5 * cm, 2.1 * cm, 2.4 * cm, 2.1 * cm, 1.8 * cm],
        ),
        p("6. Финальная модель", styles, "h1"),
        p(
            "Для сервиса выбрана TF-IDF + ручные признаки + LogisticRegression. LinearSVC был сильнее на validation, "
            "но логистическая регрессия возвращает вероятность spam-класса через predict_proba. Это удобно для API "
            "и интерфейса: пользователь видит не только класс, но и spam_score.",
            styles,
        ),
        bullet_list(
            [
                "free, call, claim, prize, urgent двигают предсказание в сторону spam;",
                "телефоны, валютные символы и CTA-слова усиливают этот сигнал;",
                "обычные бытовые SMS без цифр, ссылок и призывов чаще остаются ham.",
            ],
            styles,
        ),
        p("7. Деплой", styles, "h1"),
        p(
            "Для CP3 сделаны два локальных способа использования модели: FastAPI для программного вызова и "
            "Streamlit для ручной проверки сообщения.",
            styles,
        ),
        styled_table(
            [
                ["Компонент", "Что делает"],
                ["GET /health", "Проверяет, что API поднят"],
                ["POST /predict", "Принимает JSON с message и возвращает label, target, spam_score"],
                ["Streamlit UI", "Даёт поле ввода SMS и показывает результат модели"],
            ],
            styles,
            [4.5 * cm, 11 * cm],
        ),
        Spacer(1, 0.25 * cm),
        p("Пример ответа API:", styles),
        p('{"label": "spam", "target": 1, "spam_score": 0.9902}', styles, "code"),
    ]
    story += add_image(IMAGES_DIR / "api_predict_example.png", 13.5 * cm)
    story += add_image(IMAGES_DIR / "streamlit_interface_example.png", 13.5 * cm)
    story += [
        KeepTogether(
            [
                p("Видео демонстрации", styles, "h1"),
                p("https://disk.yandex.ru/i/pJhz8z5Rq__DAw", styles),
                p("Команды запуска:", styles),
                p(
                    "uvicorn app.api:app --host 127.0.0.1 --port 8000<br/>"
                    "streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501",
                    styles,
                    "code",
                ),
            ]
        ),
        p("8. Выводы", styles, "h1"),
        p(
            "Проект можно запустить локально и проверить на своих SMS. API подходит для программного доступа, "
            "Streamlit - для ручной проверки одного сообщения.",
            styles,
        ),
        p(
            "Главное ограничение - датасет небольшой и не новый. Для реального антиспам-фильтра следующим шагом "
            "был бы сбор свежих сообщений, проверка качества на новых данных и подбор порога spam_score под "
            "допустимый уровень ложных срабатываний.",
            styles,
        ),
    ]
    return story


def build_report_pdf() -> None:
    styles = make_styles()
    pdf_path = REPORT_DIR / "report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.9 * cm,
        title="SMS Spam Detection",
        author="Gleb Sanin",
    )
    doc.build(build_story(styles), onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_report_pdf()
