"""
image_analysis.py
High-level "describe this page / image" pipeline: renders or extracts visuals
from a PDF page and sends them to a vision-capable LLM for description, chart
data extraction, or answering a specific question about what's shown.
"""

from llm_client import analyze_image, LLMError
from pdf_vision import render_page_png, extract_embedded_images, resize_image_bytes, to_base64, PDFVisionError


def describe_page(pdf_path: str, page_number: int, question: str = None, zoom: float = 2.0) -> str:
    """Render the given page as an image and describe it (or answer a specific
    question about it) using the vision model. This is the primary path — it
    works for both real embedded images AND vector-drawn charts/diagrams that
    have no separate embedded image object.

    Raises:
        PDFVisionError: invalid page number
        LLMError: no Groq API key configured, or the vision call failed
    """
    image_bytes = render_page_png(pdf_path, page_number, zoom=zoom)
    image_bytes = resize_image_bytes(image_bytes)  # cap resolution -> cap token cost
    b64 = to_base64(image_bytes)
    prompt = None
    if question and question.strip():
        prompt = (
            f"Looking at this page from a document, answer the following question. "
            f"If the page contains a chart, graph, or table, use the actual data shown "
            f"to answer precisely, quoting specific numbers/labels where relevant.\n\n"
            f"Question: {question.strip()}"
        )
    return analyze_image(b64, mime_type="image/png", prompt=prompt)


def describe_embedded_images(pdf_path: str, page_number: int, question: str = None) -> list:
    """Extract and describe each individually embedded raster image on a page.

    Returns a list of description strings, one per image found. Returns an
    empty list if the page has no embedded raster images (common for pages
    where charts/diagrams are vector-drawn rather than embedded pictures —
    use describe_page() instead in that case, which handles both).

    Individual image analysis failures (e.g. a transient API error on one of
    several images) are captured as an error string in that image's slot
    rather than aborting the whole batch.
    """
    images = extract_embedded_images(pdf_path, page_number)
    descriptions = []
    prompt = None
    if question and question.strip():
        prompt = f"Looking at this image, answer: {question.strip()}"

    for img_bytes in images:
        img_bytes = resize_image_bytes(img_bytes)  # cap resolution -> cap token cost
        b64 = to_base64(img_bytes)
        try:
            desc = analyze_image(b64, mime_type="image/png", prompt=prompt)
        except LLMError as e:
            desc = f"(Could not analyze this image: {e})"
        descriptions.append(desc)

    return descriptions