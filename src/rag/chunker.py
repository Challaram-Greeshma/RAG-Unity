from pathlib import Path

from langchain_core.documents import Document

from rag.data_loader import docs_kind, relative_source
from rag.html_parser import extract_blocks, parse_html_article


def chunk_html_file(
    html_file: str | Path,
    data_directory: str | Path | None = None,
) -> list[Document]:
    """Parse one HTML page once and return semantic chunks."""
    html_file = Path(html_file)
    article, title = parse_html_article(html_file)
    blocks = extract_blocks(article)
    if not blocks:
        text = article.get_text("\n", strip=True)
        if not text.strip():
            return []
        blocks = [{
            "type": "article",
            "tag": "article",
            "section": title,
            "text": text,
        }]

    source_path = relative_source(html_file, data_directory)
    kind = docs_kind(html_file)
    chunks = []
    for index, block in enumerate(blocks):
        text = block["text"].strip()
        if not text:
            continue
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(html_file.resolve()),
                    "source_file": html_file.name,
                    "source_path": source_path,
                    "docs_kind": kind,
                    "file_type": "html",
                    "title": title,
                    "block_type": block["type"],
                    "tag": block["tag"],
                    "section": block["section"],
                    "block_index": index,
                },
            )
        )
    return chunks


def split_documents(
    documents: list[Document],
    data_directory: str | Path | None = None,
) -> list[Document]:
    """Chunk loaded HTML documents using structured article blocks."""
    all_chunks = []
    verbose = len(documents) < 50

    print(f"Splitting {len(documents)} documents into semantic chunks")

    for doc in documents:
        html_file = Path(doc.metadata["source"])
        if verbose:
            print(f"\nChunking: {html_file.name}")
        try:
            chunks = chunk_html_file(html_file, data_directory)
            all_chunks.extend(chunks)
            if verbose:
                print(f"  [ok] Split into {len(chunks)} chunks")
        except Exception as e:
            print(f"  [error] {html_file}: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks
