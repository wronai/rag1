import sys
from pathlib import Path

import numpy as np
import torch

import rag


class DummySentenceTransformer:
    def __init__(self, name):
        self.name = name

    def encode(self, texts, convert_to_numpy=True):
        vectors = np.ones((len(texts), 4), dtype=np.float32)
        for idx in range(len(texts)):
            vectors[idx] *= (idx + 1)
        return vectors


class DummyAutoTokenizer:
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, *args, **kwargs):
        class DummyEncoding(dict):
            def to(self, device):
                return self

        encoding = DummyEncoding({"input_ids": torch.zeros((1, 1), dtype=torch.long)})
        return encoding

    def decode(self, tokens, skip_special_tokens=True):
        return "Dummy answer"


class DummyAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def generate(self, **kwargs):
        return torch.zeros((1, 3), dtype=torch.long)


def test_main_prints_answer_with_stubbed_models(monkeypatch, tmp_path, capsys):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("Hello Bolmo", encoding="utf-8")

    monkeypatch.setenv("BOLMO_MODEL", "dummy-model")
    monkeypatch.setenv("BOLMO_EMBEDDER", "dummy-embedder")
    monkeypatch.setattr(rag, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setattr(rag, "AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr(rag, "AutoModelForCausalLM", DummyAutoModel)

    monkeypatch.setattr(sys, "argv", [
        "rag.py",
        "--folder",
        str(docs_dir),
        "--query",
        "Co to jest Bolmo?",
    ])

    rag.main()

    captured = capsys.readouterr()
    assert "Dummy answer" in captured.out


def test_main_handles_empty_folder(monkeypatch, tmp_path, capsys):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "rag.py",
        "--folder",
        str(docs_dir),
        "--query",
        "Pytanie",
    ])

    rag.main()

    captured = capsys.readouterr()
    assert "Brak dokumentów" in captured.out
