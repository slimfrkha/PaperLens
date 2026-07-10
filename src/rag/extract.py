"""PDF -> markdown extraction via Docling.

Single implementation of the Docling conversion (OCR off by default — arXiv PDFs
have a real text layer, and the OCR model download is both slow and version-fragile;
``extraction.ocr_enabled`` turns it on for scanned/no-text-layer PDFs).
Used by the ingestion pipeline (`paperlens-ingest` / the background worker).
"""

from __future__ import annotations

# Docling models are heavy — build once, reuse. `ocr_enabled` is fixed by config for
# the life of the process, so a change only ever happens in tests.
_converter = None
_converter_ocr: bool | None = None


def _get_converter(ocr_enabled: bool):
    global _converter, _converter_ocr
    if _converter is None or _converter_ocr != ocr_enabled:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = ocr_enabled
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        _converter_ocr = ocr_enabled
    return _converter


def pdf_to_markdown(pdf_path: str, *, ocr_enabled: bool = False) -> str:
    """Convert a PDF file to markdown text."""
    result = _get_converter(ocr_enabled).convert(pdf_path)
    return result.document.export_to_markdown()
