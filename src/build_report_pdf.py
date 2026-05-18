import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config import REPORT_DIR


def add_text_page(pdf: PdfPages, title: str, body: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=15, weight="bold", va="top")

    y = 0.90
    for line in body:
        font_size = 10
        if line.startswith("# "):
            wrapped = [line.removeprefix("# ")]
            font_size = 14
        elif line.startswith("## "):
            wrapped = [line.removeprefix("## ")]
            font_size = 12
        else:
            wrapped = textwrap.wrap(line, width=92) or [""]

        for part in wrapped:
            fig.text(0.08, y, part, fontsize=font_size, va="top")
            y -= 0.022 if font_size <= 10 else 0.030
            if y < 0.07:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.94

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report_pdf() -> None:
    markdown_path = REPORT_DIR / "report.md"
    pdf_path = REPORT_DIR / "report.pdf"
    lines = markdown_path.read_text(encoding="utf-8").splitlines()

    with PdfPages(pdf_path) as pdf:
        add_text_page(pdf, "SMS Spam Detection", lines)

    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_report_pdf()
