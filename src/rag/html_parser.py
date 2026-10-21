from pathlib import Path

from bs4 import BeautifulSoup


def parse_html_article(html_file: str | Path):
    """Load a Unity docs HTML file and return the cleaned article node plus title."""
    html_file = Path(html_file)
    with html_file.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    h1 = soup.find("h1")
    article = h1.find_parent("div", class_="section") if h1 else None
    if article is None:
        article = soup.find("body") or soup

    for noise in article.select(
        "div.breadcrumbs, div.nextprev, script, style, span.tooltiptext, "
        "div.suggest-success, div.suggest-failed, div.suggest-form"
    ):
        noise.decompose()

    title = h1.get_text(" ", strip=True) if h1 else html_file.stem
    return article, title


def extract_blocks(article) -> list[dict]:
    """Split an HTML article into semantic blocks."""
    blocks = []
    current_section = None
    seen_h1 = False

    elements = article.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "table"])

    for element in elements:
        if element.name in ["h1", "h2", "h3", "h4"]:
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if element.name == "h1":
                seen_h1 = True
            current_section = text
            blocks.append({
                "type": "heading",
                "tag": element.name,
                "section": current_section,
                "text": text,
            })
        elif not seen_h1:
            continue
        elif element.name == "p":
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            blocks.append({
                "type": "paragraph",
                "tag": "p",
                "section": current_section,
                "text": text,
            })
        elif element.name == "li":
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            blocks.append({
                "type": "list_item",
                "tag": "li",
                "section": current_section,
                "text": text,
            })
        elif element.name == "pre":
            text = element.get_text("\n", strip=True)
            if not text:
                continue
            blocks.append({
                "type": "code",
                "tag": "pre",
                "section": current_section,
                "text": text,
            })
        elif element.name == "table":
            rows = []
            for tr in element.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                row = [cell.get_text(" ", strip=True) for cell in cells]
                if row:
                    rows.append(row)
            if not rows:
                continue
            headers = rows[0]
            for row in rows[1:]:
                if len(row) != len(headers):
                    continue
                table_text = "\n".join(
                    f"{header}: {value}" for header, value in zip(headers, row)
                )
                blocks.append({
                    "type": "table_row",
                    "tag": "table",
                    "section": current_section,
                    "text": table_text,
                })

    return blocks
