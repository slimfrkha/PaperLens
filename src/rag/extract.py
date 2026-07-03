"""PDF -> markdown extraction via Docling.

Single implementation of the Docling conversion (OCR disabled — arXiv PDFs have
a real text layer, and the OCR model download is both slow and version-fragile).
Used by the ingestion pipeline (`paperlens-ingest` / the background worker).
"""

from __future__ import annotations

_converter = None  # Docling models are heavy — build once, reuse.


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converter


def pdf_to_markdown(pdf_path: str) -> str:
    """Convert a PDF file to markdown text."""
    result = _get_converter().convert(pdf_path)
    return result.document.export_to_markdown()
