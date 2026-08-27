"""
export_utils.py
Export chat history (questions, answers, sources) to a formatted PDF report.
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

EXPORT_DIR = Path(__file__).parent / "exports"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def export_chat_to_pdf(chat_history: list, session_id: str) -> str:
    """Render the given chat history into a PDF and return the file path."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"research_qa_{session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = EXPORT_DIR / filename

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    question_style = ParagraphStyle(
        "Question", parent=styles["Heading3"], textColor=colors.HexColor("#1a4d8f"), spaceBefore=14
    )
    answer_style = ParagraphStyle("Answer", parent=styles["Normal"], spaceBefore=6, leading=15)
    source_style = ParagraphStyle(
        "Source", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#555555"),
        leftIndent=14, spaceBefore=2,
    )
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=colors.grey)

    doc = SimpleDocTemplate(str(filepath), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = [
        Paragraph("AI Research Assistant — Q&A Export", title_style),
        Paragraph(f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", meta_style),
        Spacer(1, 16),
    ]

    for i, entry in enumerate(chat_history, start=1):
        story.append(Paragraph(f"Q{i}: {_escape(entry['question'])}", question_style))
        story.append(Paragraph(_escape(entry["answer"]).replace("\n", "<br/>"), answer_style))

        sources = entry.get("sources") or []
        if sources:
            story.append(Spacer(1, 4))
            for s in sources:
                label = f"Source: {_escape(s.get('filename', ''))}, page {s.get('page', '?')}"
                story.append(Paragraph(label, source_style))

        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#dddddd")))

    doc.build(story)
    return str(filepath)