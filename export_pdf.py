"""
Convert methodology.md → methodology.pdf using markdown + weasyprint.
Images are embedded via absolute file:// URIs so weasyprint can resolve them.
"""

import os
import re
import base64
import markdown
from weasyprint import HTML, CSS

MD_FILE  = "methodology.md"
PDF_FILE = "methodology.pdf"
BASE_DIR = os.path.abspath(".")


def embed_images(html: str) -> str:
    """Replace local image src paths with base64 data URIs."""
    def replacer(m):
        src = m.group(1)
        # skip external URLs
        if src.startswith("http"):
            return m.group(0)
        abs_path = os.path.join(BASE_DIR, src)
        if not os.path.exists(abs_path):
            return m.group(0)
        with open(abs_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'src="data:image/png;base64,{data}"'
    return re.sub(r'src="([^"]+)"', replacer, html)


def md_to_pdf(md_file: str, pdf_file: str):
    with open(md_file, encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown → HTML (tables + fenced code)
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )

    html_body = embed_images(html_body)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Methodology</title>
</head>
<body>
{html_body}
</body>
</html>"""

    css = CSS(string="""
        @page {
            size: A4;
            margin: 2.2cm 2.2cm 2.4cm 2.2cm;
            @bottom-right {
                content: counter(page);
                font-size: 9pt;
                color: #666;
            }
        }

        body {
            font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
            font-size: 10.5pt;
            line-height: 1.65;
            color: #1a1a1a;
        }

        h1 { font-size: 18pt; color: #1a3a6b; margin-top: 0; border-bottom: 2pt solid #2C6FAC; padding-bottom: 4pt; }
        h2 { font-size: 13pt; color: #2C6FAC; margin-top: 18pt; border-bottom: 1pt solid #ccd9ec; padding-bottom: 3pt; }
        h3 { font-size: 11pt; color: #3A5F8A; margin-top: 12pt; }

        p  { margin: 6pt 0; }
        blockquote {
            margin: 10pt 0;
            padding: 6pt 10pt;
            background: #f0f4fa;
            border-left: 3pt solid #2C6FAC;
            font-size: 9.5pt;
            color: #444;
        }
        blockquote p { margin: 2pt 0; }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0;
            font-size: 9.5pt;
        }
        th {
            background: #2C6FAC;
            color: white;
            padding: 5pt 7pt;
            text-align: left;
        }
        td {
            padding: 4pt 7pt;
            border-bottom: 0.5pt solid #d0d8e8;
        }
        tr:nth-child(even) td { background: #f5f8fd; }

        code {
            font-family: "DejaVu Sans Mono", monospace;
            font-size: 9pt;
            background: #f0f0f0;
            padding: 1pt 3pt;
            border-radius: 2pt;
        }
        pre code { display: block; padding: 8pt; white-space: pre-wrap; }

        img {
            max-width: 100%;
            display: block;
            margin: 10pt auto;
        }

        strong { color: #1a3a6b; }

        hr { border: none; border-top: 1pt solid #ccd9ec; margin: 16pt 0; }
    """)

    HTML(string=full_html, base_url=BASE_DIR).write_pdf(pdf_file, stylesheets=[css])
    print(f"PDF saved → {pdf_file}")


if __name__ == "__main__":
    md_to_pdf(MD_FILE, PDF_FILE)
