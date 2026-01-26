import os
from pathlib import Path

from rag import load_all_docs


def test_sample_docs_loading():
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs = load_all_docs(str(docs_dir))
    assert docs, "Sample docs directory should not be empty"

    paths = {Path(path).name for path, _ in docs}
    expected = {"readme.txt", "faq.md", "support.html", "schedule.csv"}
    assert expected.issubset(paths), "Not all expected sample documents were loaded"
