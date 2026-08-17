"""PDF -> markdown extraction via Docling.

Single implementation of the Docling conversion (OCR off by default — arXiv PDFs
have a real text layer, and the OCR model download is both slow and version-fragile;
``extraction.ocr_enabled`` turns it on for scanned/no-text-layer PDFs).
Used by the ingestion pipeline (`paperlens-ingest` / the background worker).

``render_images`` additionally crops each figure to its own image, written to a
display-only sibling markdown file + image dir — never chunked/embedded/retrieved,
purely for the paper viewer. It rides the same conversion pass (picture cropping reads
already-rasterized page regions, adding no measurable time), so there's no separate
extraction stage for it.
"""

from __future__ import annotations

import re
from pathlib import Path

# Docling models are heavy — build once, reuse. Both flags are fixed by config for the
# life of the process, so a change only ever happens in tests.
_converter = None
_converter_key: tuple[bool, bool] | None = None

# A Docling picture-crop filename, e.g. "image_000002_6f420826...f99d.png" — the hex
# hash is a content hash, so identical crops (a per-page watermark/logo, most often)
# share it.
_HASH_SUFFIX = re.compile(r"_([0-9a-f]{16,64})\.\w+$")
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _get_converter(ocr_enabled: bool, render_images: bool):
    global _converter, _converter_key
    key = (ocr_enabled, render_images)
    if _converter is None or _converter_key != key:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = ocr_enabled
        opts.generate_picture_images = render_images
        if render_images:
            opts.images_scale = 2.0  # sharp enough to read axis labels/legends
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        _converter_key = key
    return _converter


def _dedupe_images(display_md_path: Path, assets_dir: Path) -> None:
    """Docling crops every detected "picture" region, including per-page
    watermarks/logos repeated on every page — collapse those by content hash (already
    embedded in Docling's own filenames) so only the first occurrence survives, dropping
    both the duplicate file and its markdown reference. A figure intentionally repeated
    across sections would also collapse to one appearance — an accepted tradeoff.
    """
    text = display_md_path.read_text()
    seen: set[str] = set()

    def _replace(m: re.Match) -> str:
        filename = m.group(1).rsplit("/", 1)[-1]
        hash_match = _HASH_SUFFIX.search(filename)
        if not hash_match:
            return m.group(0)
        digest = hash_match.group(1)
        if digest in seen:
            (assets_dir / filename).unlink(missing_ok=True)
            return ""
        seen.add(digest)
        return m.group(0)

    deduped = _IMAGE_REF.sub(_replace, text)
    deduped = re.sub(r"\n{3,}", "\n\n", deduped)  # collapse gaps left by removed refs
    display_md_path.write_text(deduped)


def pdf_to_markdown(
    pdf_path: str,
    *,
    ocr_enabled: bool = False,
    render_images: bool = False,
    display_md_path: str | None = None,
    paper_id: str | None = None,
) -> str:
    """Convert a PDF file to markdown text.

    Returns the RAG-facing markdown (unchanged regardless of ``render_images`` — image
    placeholders stay as Docling's default ``<!-- image -->`` comment, never a real
    reference, so chunking/embedding is unaffected). When ``render_images`` is set along
    with ``display_md_path``/``paper_id``, also writes a sibling display markdown +
    ``<paper_id>.assets/`` image dir with figures cropped out and deduped by content hash.
    """
    result = _get_converter(ocr_enabled, render_images).convert(pdf_path)

    if render_images and display_md_path and paper_id:
        from docling_core.types.doc import ImageRefMode

        out_path = Path(display_md_path)
        assets_dir_name = f"{paper_id}.assets"
        # Write-temp-then-rename: a crash between save_as_markdown and _dedupe_images
        # would otherwise leave a non-empty (so "cached") but un-deduped display file
        # behind permanently — nothing else ever re-triggers extraction to fix it.
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        result.document.save_as_markdown(
            tmp_path,
            artifacts_dir=Path(assets_dir_name),  # relative: keeps refs relative to out_path
            image_mode=ImageRefMode.REFERENCED,
        )
        _dedupe_images(tmp_path, out_path.parent / assets_dir_name)
        tmp_path.replace(out_path)  # atomic on the same filesystem

    return result.document.export_to_markdown()
