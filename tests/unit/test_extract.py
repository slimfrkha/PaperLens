"""HTML normalization/fallback and Docling image-dedup helpers."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rag import extract
from rag.extract import (
    ExtractionResult,
    HTMLExtractionError,
    _dedupe_images,
    _html_result,
    extract_paper,
)

_HASH_A = "aaaa1111aaaa1111"  # 16 hex chars — matches Docling's real (64-char sha256) shape
_HASH_B = "bbbb2222bbbb2222"

_ARXIV_HTML = """<!doctype html>
<html><body>
<article class="ltx_document">
  <h1 class="ltx_title ltx_title_document">
    <img src="/html/1234.5678/figure.png" alt="[Uncaptioned image]">
    A <em>Useful</em> Paper
  </h1>
  <div class="ltx_abstract">
    <h6>Abstract</h6>
    <p>We study <math alttext="x squared">
      <annotation encoding="application/x-tex">x^2</annotation>
    </math>.</p>
  </div>
  <nav class="ltx_TOC">Table of contents noise</nav>
  <section class="ltx_section">
    <h2><span class="ltx_tag">1 </span>Introduction</h2>
    <p>See <a href="#details">the details</a>.</p>
    <ul><li><span class="ltx_tag">•</span>First contribution</li></ul>
    <figure>
      <img class="ltx_graphics" src="/html/1234.5678/figure.png" alt="Architecture">
      <figcaption>Figure 1: Architecture overview.</figcaption>
    </figure>
    <table><tr><th>Model</th><th>Score</th></tr><tr><td>A</td><td>10</td></tr></table>
    <span class="ltx_ERROR">conversion garbage</span>
    <section class="ltx_subsection" id="details">
      <h3><span class="ltx_tag">1.1 </span>Details</h3>
      <p>Nested section body.</p>
      <object class="ltx_graphics" data="/html/1234.5678/plot.svg"></object>
    </section>
  </section>
  <footer>arXiv:1234.5678v3</footer>
</article>
</body></html>
"""


def _mock_arxiv(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/html/1234.5678":
        return httpx.Response(200, text=_ARXIV_HTML, headers={"content-type": "text/html"})
    if request.url.path.endswith("figure.png"):
        return httpx.Response(200, content=b"png", headers={"content-type": "image/png"})
    if request.url.path.endswith("plot.svg"):
        return httpx.Response(200, content=b"<svg/>", headers={"content-type": "image/svg+xml"})
    return httpx.Response(404)


def test_html_result_preserves_chunker_contract_and_downloads_display_assets(tmp_path):
    html_path = tmp_path / "paper.html"
    display_path = tmp_path / "paper_display.md"
    with httpx.Client(transport=httpx.MockTransport(_mock_arxiv)) as client:
        result = _html_result(
            client,
            "1234.5678",
            "paper",
            html_path,
            display_path,
            render_images=True,
            refresh_html=False,
        )

    assert result.source == "html"
    assert result.arxiv_version == "1234.5678v3"
    assert "## A *Useful* Paper" in result.markdown
    assert "## Abstract" in result.markdown
    assert "$x^2$" in result.markdown
    assert "## 1 Introduction" in result.markdown
    assert "## 1.1 Details" in result.markdown
    assert "| Model | Score |" in result.markdown
    assert "- First contribution" in result.markdown
    assert "- •" not in result.markdown
    assert "conversion garbage" not in result.markdown
    assert "Table of contents noise" not in result.markdown
    assert "![" not in result.markdown

    assert result.display_markdown is not None
    assert result.display_markdown.splitlines()[0] == "## A *Useful* Paper"
    assert "paper.assets/" in result.display_markdown
    assert display_path.read_text() == result.display_markdown
    assert html_path.read_text() == _ARXIV_HTML
    assets = list((tmp_path / "paper.assets").iterdir())
    assert sorted(path.suffix for path in assets) == [".png", ".svg"]


def test_real_latexml_sample_matches_golden_markdown(tmp_path):
    """Reduced markup captured from arXiv:2606.13392v2; no network in CI."""
    data_dir = Path(__file__).parents[1] / "data"
    html_path = tmp_path / "paper.html"
    html_path.write_text((data_dir / "arxiv_latexml_sample.html").read_text())

    with httpx.Client(transport=httpx.MockTransport(_mock_arxiv)) as client:
        result = _html_result(
            client,
            "2606.13392",
            "paper",
            html_path,
            tmp_path / "paper_display.md",
            render_images=False,
            refresh_html=False,
        )

    assert result.markdown == (data_dir / "arxiv_latexml_sample.md").read_text()


def test_html_result_rejects_non_paper_html(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>not a paper</html>", headers={"content-type": "text/html"}
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(HTMLExtractionError, match="article.ltx_document"),
    ):
        _html_result(
            client,
            "1234.5678",
            "paper",
            tmp_path / "paper.html",
            tmp_path / "paper_display.md",
            render_images=False,
            refresh_html=False,
        )


def test_auto_backend_falls_back_to_docling_and_records_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(
        extract,
        "_html_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(HTMLExtractionError("no HTML")),
    )
    fallback = ExtractionResult(
        markdown="## PDF title\n",
        display_markdown=None,
        source="docling",
        source_url="https://arxiv.org/pdf/1234.5678",
        arxiv_version="1234.5678",
    )
    monkeypatch.setattr(extract, "_docling_result", lambda *args, **kwargs: fallback)

    result = extract_paper(
        "1234.5678",
        "paper",
        pdf_path=str(tmp_path / "paper.pdf"),
        html_path=str(tmp_path / "paper.html"),
        display_md_path=str(tmp_path / "paper_display.md"),
    )

    assert result.source == "docling"
    assert result.warnings == ("HTML unavailable; used Docling: no HTML",)


def test_html_backend_is_strict_and_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(
        extract,
        "_html_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(HTMLExtractionError("no HTML")),
    )
    monkeypatch.setattr(
        extract,
        "_docling_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not fall back")),
    )

    with pytest.raises(HTMLExtractionError, match="no HTML"):
        extract_paper(
            "1234.5678",
            "paper",
            pdf_path=str(tmp_path / "paper.pdf"),
            html_path=str(tmp_path / "paper.html"),
            display_md_path=str(tmp_path / "paper_display.md"),
            backend="html",
        )


def test_dedupe_images_drops_repeated_hash_keeps_first(tmp_path):
    # Same content hash (a per-page watermark, most often) referenced three times;
    # a distinct hash referenced once.
    md = (
        "intro\n\n"
        f"![Image](p.assets/image_000000_{_HASH_A}.png)\n\n"
        "middle\n\n"
        f"![Image](p.assets/image_000001_{_HASH_B}.png)\n\n"
        f"![Image](p.assets/image_000002_{_HASH_A}.png)\n\n"
        "end"
    )
    display_md = tmp_path / "p_display.md"
    display_md.write_text(md)
    assets_dir = tmp_path / "p.assets"
    assets_dir.mkdir()
    (assets_dir / f"image_000000_{_HASH_A}.png").write_bytes(b"x")
    (assets_dir / f"image_000001_{_HASH_B}.png").write_bytes(b"y")
    (assets_dir / f"image_000002_{_HASH_A}.png").write_bytes(b"x")

    _dedupe_images(display_md, assets_dir)

    out = display_md.read_text()
    assert out.count(f"image_000000_{_HASH_A}.png") == 1  # first occurrence survives
    assert f"image_000002_{_HASH_A}.png" not in out  # duplicate reference stripped
    assert f"image_000001_{_HASH_B}.png" in out  # distinct hash untouched
    assert (assets_dir / f"image_000000_{_HASH_A}.png").exists()
    assert not (assets_dir / f"image_000002_{_HASH_A}.png").exists()  # duplicate file removed
    assert (assets_dir / f"image_000001_{_HASH_B}.png").exists()


def test_dedupe_images_noop_when_no_duplicates(tmp_path):
    md = (
        f"![Image](p.assets/image_000000_{_HASH_A}.png)\n\n"
        f"![Image](p.assets/image_000001_{_HASH_B}.png)"
    )
    display_md = tmp_path / "p_display.md"
    display_md.write_text(md)
    assets_dir = tmp_path / "p.assets"
    assets_dir.mkdir()
    (assets_dir / f"image_000000_{_HASH_A}.png").write_bytes(b"x")
    (assets_dir / f"image_000001_{_HASH_B}.png").write_bytes(b"y")

    _dedupe_images(display_md, assets_dir)

    assert display_md.read_text() == md
    assert len(list(assets_dir.iterdir())) == 2


def test_dedupe_images_leaves_non_docling_image_refs_alone(tmp_path):
    # A markdown image ref that doesn't match Docling's `..._<hash>.ext` naming (e.g.
    # hand-authored content) has no hash to key on — leave it untouched rather than guess.
    md = "![a diagram](some/other/path.png)"
    display_md = tmp_path / "p_display.md"
    display_md.write_text(md)
    assets_dir = tmp_path / "p.assets"
    assets_dir.mkdir()

    _dedupe_images(display_md, assets_dir)

    assert display_md.read_text() == md
