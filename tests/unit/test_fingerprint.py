"""The eval corpus contains canonical markdown only, never viewer artifacts."""

from eval.fingerprint import load_pool


def test_load_pool_ignores_display_markdown(tmp_path):
    (tmp_path / "paper-a.md").write_text("canonical")
    (tmp_path / "paper-a_display.md").write_text("viewer copy with images")

    assert load_pool(str(tmp_path)) == {"paper-a": "canonical"}
