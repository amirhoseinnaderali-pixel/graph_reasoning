import json
from pathlib import Path

from scripts.analyze_graph_structure import entropy


def test_entropy_uniform():
    assert round(entropy(["a", "b", "c", "d"]), 6) == 2.0


def test_entropy_constant():
    assert entropy(["a", "a", "a"]) == 0.0


def test_existing_artifact_exists():
    assert Path("got_data.json").exists()
