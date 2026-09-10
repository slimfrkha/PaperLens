"""arXiv HTML -> markdown extraction with PDF/Docling fallback.

The normal path fetches arXiv's source-derived HTML, preserves its semantic sections,
math, tables, links, and figure assets, then emits the same two markdown artifacts the
rest of PaperLens already consumes. When HTML is unavailable or structurally invalid,
``auto`` mode falls back to the existing PDF/Docling conversion.

Docling remains lazy: a successful HTML extraction never imports or initializes it.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

ARXIV_HTML = "https://arxiv.org/html/{id}"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
_USER_AGENT = "PaperLens/0.1 (local research paper ingestion)"
_MAX_ASSET_BYTES = 25 * 1024 * 1024
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_CLASSES = {"ltx_TOC", "ltx_page_footer", "ltx_role_dubious"}


class HTMLExtractionError(RuntimeError):
    """arXiv HTML was unavailable or did not contain a usable paper."""


@dataclass(frozen=True)
class ExtractionResult:
    markdown: str
    display_markdown: str | None
    source: str
    source_url: str
    arxiv_version: str
    warnings: tuple[str, ...] = ()


# Docling models are heavy — build once, reuse. Both flags are fixed by config for the
# life of the process, so a change only ever happens in tests.
_converter = None
_converter_key: tuple[bool, bool] | None = None

# A Docling picture-crop filename, e.g. "image_000002_6f420826...f99d.png" — the hex
# hash is a content hash, so identical crops (a per-page watermark/logo, most often)
# share it.
_HASH_SUFFIX = re.compile(r"_([0-9a-f]{16,64})\.\w+$")
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _classes(node: Tag) -> set[str]:
    value = node.get("class")
    if value is None:
        return set()
    return set(value if isinstance(value, list) else str(value).split())


def _clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _MarkdownRenderer:
    """Small LaTeXML-aware renderer, deliberately scoped to arXiv paper HTML."""

    def __init__(self, base_url: str, assets: dict[str, str], *, display: bool):
        self.base_url = base_url
        self.assets = assets
        self.display = display

    def _children_inline(self, node: Tag) -> str:
        return "".join(self.inline(child) for child in node.children)

    def _heading_text(self, node: Tag) -> str:
        """Render heading text without decorative images, including in display markdown."""
        renderer = self if not self.display else _MarkdownRenderer(self.base_url, {}, display=False)
        return _clean_inline(renderer._children_inline(node))

    def inline(self, node) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        classes = _classes(node)
        if classes & _SKIP_CLASSES or "ltx_ERROR" in classes:
            return ""

        if "ltx_note_mark" in classes or "ltx_note_type" in classes:
            return ""
        if "ltx_note" in classes:
            content = node.select_one(".ltx_note_content")
            label = _clean_inline(self._children_inline(content or node))
            return f" ({label})" if label else ""

        name = node.name.lower()
        if name == "math":
            annotation = node.find("annotation", attrs={"encoding": "application/x-tex"})
            tex = annotation.get_text(strip=True) if annotation else ""
            if tex:
                delimiter = "$$" if node.get("display") == "block" else "$"
                return f"{delimiter}{tex}{delimiter}"
            return _clean_inline(str(node.get("alttext", ""))) or _clean_inline(
                node.get_text(" ", strip=True)
            )
        if name in {"img", "object"}:
            if not self.display:
                return ""
            raw = node.get("src") if name == "img" else node.get("data")
            if not raw:
                return ""
            resolved = urljoin(self.base_url, str(raw))
            local = self.assets.get(resolved)
            if not local:
                return ""
            alt = _clean_inline(str(node.get("alt", "Figure"))).replace("[", "").replace("]", "")
            alt = alt or "Figure"
            return f"![{alt}]({local})"
        if name == "a":
            label = _clean_inline(self._children_inline(node))
            href = str(node.get("href", ""))
            if not label or not href or href.startswith(("#", "javascript:")):
                return label
            return f"[{label}]({urljoin(self.base_url, href)})"
        if name == "br":
            return "\n"
        if name == "code":
            return f"`{node.get_text()}`"
        if name in {"strong", "b"}:
            return f"**{_clean_inline(self._children_inline(node))}**"
        if name in {"em", "i"}:
            return f"*{_clean_inline(self._children_inline(node))}*"
        if name == "sup":
            return f"^({_clean_inline(self._children_inline(node))})"
        if name == "sub":
            return f"_({_clean_inline(self._children_inline(node))})"
        return self._children_inline(node)

    def _table(self, table: Tag) -> str:
        if any("equation" in cls for cls in _classes(table)):
            rows = []
            for tr in table.find_all("tr"):
                value = _clean_inline(" ".join(self.inline(c) for c in tr.find_all(["th", "td"])))
                if value:
                    rows.append(value)
            return "\n\n".join(rows)

        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [
                _clean_inline(self._children_inline(cell)).replace("|", "\\|")
                for cell in tr.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        lines = ["| " + " | ".join(row) + " |" for row in rows]
        lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
        return "\n".join(lines)

    def _list(self, node: Tag, *, ordered: bool) -> str:
        lines = []
        for i, item in enumerate(node.find_all("li", recursive=False), 1):
            body = _clean_inline(
                "".join(
                    self.inline(child)
                    for child in item.children
                    if not (isinstance(child, Tag) and "ltx_tag" in _classes(child))
                )
            )
            if body:
                lines.append(f"{i}. {body}" if ordered else f"- {body}")
        return "\n".join(lines)

    def block(self, node) -> str:
        if isinstance(node, NavigableString):
            return _clean_inline(str(node))
        if not isinstance(node, Tag):
            return ""
        classes = _classes(node)
        if classes & _SKIP_CLASSES or "ltx_ERROR" in classes:
            return ""
        if node.name in {"script", "style", "nav"}:
            return ""
        if node.name == "section":
            return ""  # semantic sections are emitted by document() below
        if node.name in _HEADINGS:
            title = self._heading_text(node)
            return f"**{title}**" if title else ""
        if node.name == "table":
            return self._table(node)
        if node.name == "pre":
            return f"```\n{node.get_text().strip()}\n```"
        if node.name == "ul":
            return self._list(node, ordered=False)
        if node.name == "ol":
            return self._list(node, ordered=True)
        if node.name in {"p", "figcaption"}:
            return _clean_inline(self._children_inline(node))
        if node.name in {"img", "object"}:
            return self.inline(node)

        parts = [self.block(child) for child in node.children]
        return "\n\n".join(part for part in parts if part)

    def _section(self, section: Tag) -> list[str]:
        direct = [child for child in section.children if isinstance(child, Tag)]
        heading = next((child for child in direct if child.name in _HEADINGS), None)
        parts: list[str] = []
        if heading is not None:
            title = self._heading_text(heading)
            if title:
                parts.append(f"## {title}")
        body = [
            self.block(child)
            for child in direct
            if child is not heading and child.name != "section"
        ]
        parts.extend(part for part in body if part)
        for child in direct:
            if child.name == "section":
                parts.extend(self._section(child))
        return parts

    def document(self, article: Tag) -> tuple[str, str]:
        title_node = article.select_one("h1.ltx_title_document")
        if title_node is None:
            raise HTMLExtractionError("HTML paper has no document title")
        title = self._heading_text(title_node)
        if not title:
            raise HTMLExtractionError("HTML paper has an empty document title")

        parts = [f"## {title}"]
        abstract = article.select_one(".ltx_abstract")
        if abstract is not None:
            abstract_body = "\n\n".join(
                part
                for child in abstract.children
                if isinstance(child, Tag) and child.name not in _HEADINGS
                if (part := self.block(child))
            )
            if abstract_body:
                parts.extend(["## Abstract", abstract_body])

        if self.display:
            for child in article.children:
                if not isinstance(child, Tag):
                    continue
                if child is title_node or child is abstract or child.name in {"section", "nav"}:
                    continue
                if value := self.block(child):
                    parts.append(value)

        sections = [
            child
            for child in article.children
            if isinstance(child, Tag) and child.name == "section"
        ]
        for section in sections:
            parts.extend(self._section(section))
        if not sections:
            raise HTMLExtractionError("HTML paper has no semantic sections")
        if not any(part.strip() for part in parts if not part.startswith("## ")):
            raise HTMLExtractionError("HTML paper has no extractable body text")
        return title, re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip() + "\n"


def _parse_html(html: str) -> tuple[BeautifulSoup, Tag, int]:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.ltx_document")
    if not isinstance(article, Tag):
        raise HTMLExtractionError("response has no article.ltx_document")
    return soup, article, len(article.select(".ltx_ERROR"))


def _version(html: str, arxiv_id: str) -> str:
    match = re.search(rf"arXiv:{re.escape(arxiv_id)}(v\d+)", html)
    return f"{arxiv_id}{match.group(1)}" if match else arxiv_id


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _asset_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return suffix
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
    }.get(content_type.split(";", 1)[0].lower(), ".img")


def _download_assets(
    client: httpx.Client, article: Tag, base_url: str, assets_dir: Path, paper_id: str
) -> tuple[dict[str, str], list[str]]:
    urls: list[str] = []
    for node in article.select("img.ltx_graphics, object.ltx_graphics"):
        raw = node.get("src") if node.name == "img" else node.get("data")
        if raw:
            resolved = urljoin(base_url, str(raw))
            if resolved not in urls:
                urls.append(resolved)

    tmp_dir = assets_dir.with_name(assets_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mapped: dict[str, str] = {}
    warnings: list[str] = []
    for i, url in enumerate(urls):
        if urlparse(url).hostname != "arxiv.org":
            warnings.append(f"skipped non-arXiv figure URL: {url}")
            continue
        dest: Path | None = None
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                extension = _asset_extension(url, response.headers.get("content-type", ""))
                digest = hashlib.sha256(url.encode()).hexdigest()[:12]
                filename = f"image_{i:06d}_{digest}{extension}"
                dest = tmp_dir / filename
                size = 0
                with dest.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_ASSET_BYTES:
                            raise HTMLExtractionError(f"figure exceeds {_MAX_ASSET_BYTES} bytes")
                        output.write(chunk)
                mapped[url] = f"{paper_id}.assets/{filename}"
        except Exception as exc:
            if dest is not None:
                dest.unlink(missing_ok=True)
            warnings.append(f"figure download failed for {url}: {exc}")

    shutil.rmtree(assets_dir, ignore_errors=True)
    tmp_dir.replace(assets_dir)
    return mapped, warnings


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


def _download_pdf(arxiv_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with httpx.stream(
            "GET",
            ARXIV_PDF.format(id=arxiv_id),
            follow_redirects=True,
            timeout=120,
            headers={"User-Agent": _USER_AGENT},
        ) as response:
            response.raise_for_status()
            with tmp.open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _html_result(
    client: httpx.Client,
    arxiv_id: str,
    paper_id: str,
    html_path: Path,
    display_md_path: Path,
    *,
    render_images: bool,
    refresh_html: bool,
) -> ExtractionResult:
    url = ARXIV_HTML.format(id=arxiv_id)
    warnings: list[str] = []
    html: str
    if html_path.exists() and html_path.stat().st_size > 0 and not refresh_html:
        html = html_path.read_text()
    else:
        try:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                raise HTMLExtractionError(f"unexpected content type: {content_type or '(missing)'}")
            html = response.text
        except Exception as exc:
            if not (refresh_html and html_path.exists() and html_path.stat().st_size > 0):
                raise HTMLExtractionError(f"HTML fetch failed: {exc}") from exc
            html = html_path.read_text()
            warnings.append(f"HTML refresh failed; reused cached HTML: {exc}")

    _, article, error_count = _parse_html(html)
    if error_count:
        warnings.append(f"arXiv HTML contains {error_count} LaTeXML error node(s)")

    # Validate and build the canonical corpus text before touching any display assets.
    _, markdown = _MarkdownRenderer(url, {}, display=False).document(article)
    assets: dict[str, str] = {}
    if render_images:
        assets, asset_warnings = _download_assets(
            client,
            article,
            url,
            display_md_path.with_name(f"{paper_id}.assets"),
            paper_id,
        )
        warnings.extend(asset_warnings)
    else:
        display_md_path.unlink(missing_ok=True)
        shutil.rmtree(display_md_path.with_name(f"{paper_id}.assets"), ignore_errors=True)

    display_markdown = (
        _MarkdownRenderer(url, assets, display=True).document(article)[1] if render_images else None
    )
    _atomic_write(html_path, html)
    if display_markdown is not None:
        _atomic_write(display_md_path, display_markdown)
    return ExtractionResult(
        markdown=markdown,
        display_markdown=display_markdown,
        source="html",
        source_url=url,
        arxiv_version=_version(html, arxiv_id),
        warnings=tuple(warnings),
    )


def _docling_result(
    arxiv_id: str,
    paper_id: str,
    pdf_path: Path,
    display_md_path: Path,
    *,
    ocr_enabled: bool,
    render_images: bool,
) -> ExtractionResult:
    _download_pdf(arxiv_id, pdf_path)
    if not render_images:
        display_md_path.unlink(missing_ok=True)
        shutil.rmtree(display_md_path.with_name(f"{paper_id}.assets"), ignore_errors=True)
    markdown = pdf_to_markdown(
        str(pdf_path),
        ocr_enabled=ocr_enabled,
        render_images=render_images,
        display_md_path=str(display_md_path) if render_images else None,
        paper_id=paper_id if render_images else None,
    )
    return ExtractionResult(
        markdown=markdown,
        display_markdown=display_md_path.read_text() if render_images else None,
        source="docling",
        source_url=ARXIV_PDF.format(id=arxiv_id),
        arxiv_version=arxiv_id,
    )


def extract_paper(
    arxiv_id: str,
    paper_id: str,
    *,
    pdf_path: str,
    html_path: str,
    display_md_path: str,
    backend: str = "auto",
    ocr_enabled: bool = False,
    render_images: bool = True,
    refresh_html: bool = False,
) -> ExtractionResult:
    """Extract one paper through HTML, or PDF/Docling when configured/required."""
    pdf = Path(pdf_path)
    html = Path(html_path)
    display = Path(display_md_path)
    html_error: HTMLExtractionError | None = None

    if backend != "docling":
        with httpx.Client(
            follow_redirects=True,
            timeout=120,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            try:
                return _html_result(
                    client,
                    arxiv_id,
                    paper_id,
                    html,
                    display,
                    render_images=render_images,
                    refresh_html=refresh_html,
                )
            except Exception as exc:
                html_error = (
                    exc
                    if isinstance(exc, HTMLExtractionError)
                    else HTMLExtractionError(f"HTML normalization failed: {exc}")
                )
        if backend == "html":
            raise html_error

    result = _docling_result(
        arxiv_id,
        paper_id,
        pdf,
        display,
        ocr_enabled=ocr_enabled,
        render_images=render_images,
    )
    if html_error is None:
        return result
    return ExtractionResult(
        markdown=result.markdown,
        display_markdown=result.display_markdown,
        source=result.source,
        source_url=result.source_url,
        arxiv_version=result.arxiv_version,
        warnings=(f"HTML unavailable; used Docling: {html_error}",),
    )
