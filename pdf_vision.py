"""
pdf_vision.py
Extracts visual content from PDFs for analysis by a vision-capable LLM.

Two extraction modes:
    render_page_png()      — renders an entire page as an image. This is the
                              primary/most reliable path, since charts and
                              diagrams are very often drawn as vector graphics
                              with no separate embedded image object; the only
                              reliable way to "see" them is to render the whole
                              page as a picture.
    extract_embedded_images() — pulls out individual embedded raster images
                              (actual photos/figures) from a page. Returns an
                              empty list for pages with only vector-drawn
                              content (e.g. a matplotlib-style chart) — use
                              render_page_png() for those instead.

resize_image_bytes() caps image dimensions before sending to a vision API,
since vision model token cost scales with image resolution — this matters a
lot on Groq's free tier, where the rate limit is a tight 8000 tokens/minute
and a single large, uncompressed image can eat a big chunk of that budget.
"""

import base64
import io

import fitz  # PyMuPDF
from PIL import Image

MAX_IMAGE_DIMENSION = 1024  # px, caps vision API token cost


class PDFVisionError(Exception):
    pass


def resize_image_bytes(image_bytes: bytes, max_dimension: int = MAX_IMAGE_DIMENSION) -> bytes:
    """Downscale an image so its longest side is at most max_dimension pixels,
    re-encoded as PNG. This is applied right before sending any image to the
    vision API — whether it's a rendered PDF page, an extracted embedded image,
    or a raw uploaded photo — so token cost stays predictable regardless of
    source resolution. Charts and text stay readable at this size; a 1024px
    chart is still perfectly legible to a vision model."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    width, height = img.size
    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / longest_side
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def get_page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def render_page_png(pdf_path: str, page_number: int, zoom: float = 2.0) -> bytes:
    """Render a single page (1-indexed) to PNG bytes at the given zoom factor.
    zoom=2.0 is roughly 144 DPI — a good balance of clarity vs file size for
    vision models (Groq's limit is 20MB per image; a rendered page at this
    zoom is typically well under 1MB)."""
    with fitz.open(pdf_path) as doc:
        if not (1 <= page_number <= doc.page_count):
            raise PDFVisionError(f"Page {page_number} is out of range (1-{doc.page_count}).")
        page = doc.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")


def extract_embedded_images(pdf_path: str, page_number: int) -> list:
    """Extract raw embedded raster images from a single page (1-indexed), each
    normalized to PNG bytes. Images that fail to extract cleanly (unusual color
    spaces, corrupt streams, etc.) are silently skipped rather than raising,
    since a single bad image shouldn't block extraction of the rest."""
    images = []
    with fitz.open(pdf_path) as doc:
        if not (1 <= page_number <= doc.page_count):
            raise PDFVisionError(f"Page {page_number} is out of range (1-{doc.page_count}).")
        page = doc.load_page(page_number - 1)
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png").lower()
                if ext not in ("png", "jpg", "jpeg"):
                    # normalize unusual formats (e.g. CMYK, unusual color spaces)
                    # to PNG by re-rendering through a Pixmap
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes("png")
                images.append(image_bytes)
            except Exception:
                continue
    return images


def to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")