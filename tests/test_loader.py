from rag.chunker import split_documents
from rag.data_loader import load_html_file, process_all_data
from rag.html_parser import extract_blocks, parse_html_article

EXPECTED_SAMPLE_FILES = {
    "2d-game-creation-wokflow.html",
    "2DAnd3DModeSettings.html",
    "2d-renderer-sorting.html",
}


def test_load_html_file_extracts_title_and_text(sample_data_dir):
    html_file = sample_data_dir / "2DAnd3DModeSettings.html"
    document = load_html_file(html_file)

    assert document.metadata["source_file"] == "2DAnd3DModeSettings.html"
    assert document.metadata["file_type"] == "html"
    assert "2D and 3D mode settings" in document.metadata["title"]
    assert "switch" in document.page_content.lower()
    assert "2D" in document.page_content


def test_parse_html_article_uses_section_div(sample_data_dir):
    article, title = parse_html_article(
        sample_data_dir / "2d-game-creation-wokflow.html"
    )
    assert "workflow" in title.lower() or "2D" in title
    assert article.name in {"div", "body", "[document]"} or "section" in article.get(
        "class", []
    )


def test_extract_blocks_includes_scripting_section(sample_data_dir):
    article, _title = parse_html_article(
        sample_data_dir / "2d-game-creation-wokflow.html"
    )
    blocks = extract_blocks(article)
    texts = [block["text"].lower() for block in blocks]

    assert blocks
    assert any(block["type"] == "heading" for block in blocks)
    assert any("script" in text for text in texts)


def test_process_all_data_loads_sample_files(sample_data_dir):
    documents = process_all_data(sample_data_dir)
    names = {doc.metadata["source_file"] for doc in documents}

    assert names == EXPECTED_SAMPLE_FILES


def test_split_documents_preserves_source_and_block_index(sample_data_dir):
    documents = process_all_data(sample_data_dir)
    chunks = split_documents(documents)

    assert len(chunks) > len(documents)
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert {chunk.metadata["source_file"] for chunk in chunks} == EXPECTED_SAMPLE_FILES
    assert all("block_index" in chunk.metadata for chunk in chunks)
    assert all("section" in chunk.metadata for chunk in chunks)
