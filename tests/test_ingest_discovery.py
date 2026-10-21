from pathlib import Path
from types import SimpleNamespace

from rag.chunker import chunk_html_file
from rag.config import DOCS_DIR, default_ingest_dir
from rag.data_loader import iter_html_files
from rag.generator import RAGGenerator


ARTICLE_HTML = """<html><body>
<div class="section">
<h1>Title</h1>
<p>Hello Unity.</p>
</div>
</body></html>
"""


def _write_html(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ARTICLE_HTML, encoding="utf-8")


def test_iter_html_files_indexes_manual_and_script(tmp_path):
    _write_html(tmp_path / "Manual" / "Foo.html")
    _write_html(tmp_path / "ScriptReference" / "Bar.html")
    _write_html(tmp_path / "StaticFiles" / "skip.html")
    _write_html(tmp_path / "Manual" / "30_search.html")
    _write_html(tmp_path / "uploads" / "img.html")

    names = {path.name for path in iter_html_files(tmp_path)}
    assert names == {"Foo.html", "Bar.html"}


def test_iter_html_files_scope_manual(tmp_path):
    _write_html(tmp_path / "Manual" / "Foo.html")
    _write_html(tmp_path / "ScriptReference" / "Bar.html")

    names = {path.name for path in iter_html_files(tmp_path, scope="manual")}
    assert names == {"Foo.html"}


def test_iter_html_files_keeps_flat_sample_dir(tmp_path):
    _write_html(tmp_path / "2DAnd3DModeSettings.html")
    _write_html(tmp_path / "2d-renderer-sorting.html")

    names = {path.name for path in iter_html_files(tmp_path)}
    assert names == {"2DAnd3DModeSettings.html", "2d-renderer-sorting.html"}


def test_chunk_ids_include_folder_prefix(tmp_path):
    _write_html(tmp_path / "Manual" / "Shared.html")
    _write_html(tmp_path / "ScriptReference" / "Shared.html")

    manual = chunk_html_file(tmp_path / "Manual" / "Shared.html", tmp_path)
    script = chunk_html_file(tmp_path / "ScriptReference" / "Shared.html", tmp_path)

    assert manual[0].metadata["source_path"] == "Manual/Shared.html"
    assert script[0].metadata["source_path"] == "ScriptReference/Shared.html"
    assert manual[0].metadata["docs_kind"] == "Manual"
    assert script[0].metadata["docs_kind"] == "ScriptReference"


def test_script_reference_url_fallback():
    generator = RAGGenerator(
        retriever=SimpleNamespace(),
        llm=SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content="")),
    )
    url = generator._page_url({
        "source": "missing.html",
        "source_file": "MonoBehaviour.html",
        "source_path": "ScriptReference/MonoBehaviour.html",
        "docs_kind": "ScriptReference",
    })
    assert url.endswith("/ScriptReference/MonoBehaviour.html")


def test_default_ingest_dir_prefers_documentation_dump():
    ingest_dir = default_ingest_dir()
    if (DOCS_DIR / "Manual").is_dir():
        assert ingest_dir == DOCS_DIR
    else:
        assert ingest_dir.name == "data"
