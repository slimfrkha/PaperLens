"""Image-dedup post-processing (no Docling/model calls — pure text/file manipulation)."""

from __future__ import annotations

from rag.extract import _dedupe_images

_HASH_A = "aaaa1111aaaa1111"  # 16 hex chars — matches Docling's real (64-char sha256) shape
_HASH_B = "bbbb2222bbbb2222"


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
