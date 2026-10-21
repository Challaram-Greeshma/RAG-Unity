from pathlib import Path

from langchain_core.documents import Document

from rag.config import DATA_DIR
from rag.html_parser import parse_html_article

SKIP_HTML_NAMES = {"30_search.html"}
SKIP_DIR_NAMES = {"StaticFiles", "StaticFilesManual", "uploads", "docdata"}


def docs_kind(html_file: str | Path) -> str:
    parts = Path(html_file).parts
    if "ScriptReference" in parts:
        return "ScriptReference"
    return "Manual"


def relative_source(html_file: str | Path, data_directory: str | Path | None = None) -> str:
    html_file = Path(html_file).resolve()
    if data_directory is not None:
        try:
            return html_file.relative_to(Path(data_directory).resolve()).as_posix()
        except ValueError:
            pass
    return html_file.name


def _article_roots(data_dir: Path, scope: str = "all") -> list[Path]:
    manual_candidates = [data_dir / "en" / "Manual", data_dir / "Manual"]
    script_candidates = [data_dir / "en" / "ScriptReference", data_dir / "ScriptReference"]

    roots: list[Path] = []
    if scope in {"all", "manual"}:
        roots.extend(path for path in manual_candidates if path.is_dir())
    if scope in {"all", "script"}:
        roots.extend(path for path in script_candidates if path.is_dir())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique or [data_dir]


def iter_html_files(data_directory: str | Path | None = None, scope: str = "all"):
    """Yield Unity article HTML files, skipping search pages and static assets."""
    data_dir = Path(data_directory) if data_directory is not None else DATA_DIR
    for root in _article_roots(data_dir, scope=scope):
        for html_file in sorted(root.rglob("*.html")):
            if html_file.name in SKIP_HTML_NAMES:
                continue
            if any(part in SKIP_DIR_NAMES for part in html_file.parts):
                continue
            yield html_file


def load_html_file(
    html_file: str | Path,
    data_directory: str | Path | None = None,
) -> Document:
    """Parse a Unity docs HTML file into a LangChain Document."""
    html_file = Path(html_file)
    article, title = parse_html_article(html_file)
    text = article.get_text("\n", strip=True)

    if not text.strip():
        raise ValueError("No article text extracted")

    return Document(
        page_content=text,
        metadata={
            "source": str(html_file.resolve()),
            "source_file": html_file.name,
            "source_path": relative_source(html_file, data_directory),
            "docs_kind": docs_kind(html_file),
            "file_type": "html",
            "title": title,
        },
    )


def process_all_data(
    data_directory: str | Path | None = None,
    scope: str = "all",
) -> list[Document]:
    """Process all HTML files in a directory."""
    data_dir = Path(data_directory) if data_directory is not None else DATA_DIR
    html_files = list(iter_html_files(data_dir, scope=scope))
    all_documents = []
    verbose = len(html_files) < 50

    print(f"Found {len(html_files)} HTML files to process")

    for index, html_file in enumerate(html_files, start=1):
        if verbose:
            print(f"\nProcessing: {html_file.name}")
        try:
            all_documents.append(load_html_file(html_file, data_dir))
            if verbose:
                print("  [ok] Loaded 1 document")
        except Exception as e:
            print(f"  [error] {html_file}: {e}")
        if not verbose and index % 100 == 0:
            print(f"Loaded {index}/{len(html_files)} pages")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
