import json
from pathlib import Path

import pytest

from eva_agent.adapters.inputs.dataset import DatasetError, write_dataset_workspace
from eva_agent.adapters.inputs.manifest import canonical_json, sha256_bytes, write_manifest
from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.domain.evidence import EvidenceFile
from eva_agent.domain.run import DatasetRef, InputKind, PastedInput


def test_canonical_json_is_stable() -> None:
    first = canonical_json({"b": 1, "a": "中文"})
    second = canonical_json({"a": "中文", "b": 1})

    assert first == second
    assert sha256_bytes(first) == sha256_bytes(second)


def test_write_manifest_hashes_structure(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    files = [EvidenceFile(path="dataset.csv", size=3, sha256="a" * 64)]

    digest = write_manifest(
        evidence_dir,
        kind=InputKind.DATASET,
        files=files,
        pasted=None,
        dataset=DatasetRef(path="dataset.csv", format="csv"),
    )

    assert len(digest) == 64
    payload = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "dataset"
    assert payload["files"][0]["path"] == "dataset.csv"
    assert payload["dataset"]["path"] == "dataset.csv"


def test_paste_workspace_writes_paste_json_and_manifest(tmp_path: Path) -> None:
    bundle = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)

    evidence_dir = tmp_path / "runs" / "run-1" / "evidence"
    assert (evidence_dir / "paste.json").is_file()
    assert (evidence_dir / "manifest.json").is_file()
    assert bundle.kind is InputKind.PASTE
    assert bundle.root == str(evidence_dir)
    assert bundle.pasted == {"query": "q", "response": "a"}
    assert bundle.files[0].sha256 == sha256_bytes((evidence_dir / "paste.json").read_bytes())


def test_paste_workspace_hash_is_deterministic(tmp_path: Path) -> None:
    first = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)
    second = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)

    assert first.manifest_sha256 == second.manifest_sha256


def test_dataset_workspace_copies_and_rewrites_path(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("query,response,reference\nq,a,r\n", encoding="utf-8")

    bundle = write_dataset_workspace(
        "run-2",
        DatasetRef(path=str(source), format="csv"),
        tmp_path / "data",
        max_bytes=10_000,
        max_rows=10,
    )

    evidence_dir = tmp_path / "data" / "runs" / "run-2" / "evidence"
    copied = evidence_dir / "dataset.csv"
    assert copied.is_file()
    assert bundle.dataset is not None
    assert bundle.dataset.path == "dataset.csv"
    assert bundle.root == str(evidence_dir)
    assert bundle.files[0].sha256 == sha256_bytes(copied.read_bytes())


def test_dataset_workspace_rejects_bad_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("query,response\nq,\n", encoding="utf-8")

    with pytest.raises(DatasetError):
        write_dataset_workspace(
            "run-3",
            DatasetRef(path=str(source), format="csv"),
            tmp_path / "data",
            max_bytes=10_000,
            max_rows=10,
        )

    copied = tmp_path / "data" / "runs" / "run-3" / "evidence" / "dataset.csv"
    assert not copied.exists()


def test_paste_workspace_rejects_path_like_run_id(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="invalid run_id"):
        write_paste_workspace("../../escape", PastedInput(query="q"), data_dir)

    assert not (tmp_path / "escape").exists()
    assert not data_dir.exists()


def test_dataset_workspace_rejects_path_like_run_id(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("query,response\nq,a\n", encoding="utf-8")
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="invalid run_id"):
        write_dataset_workspace(
            "../../escape",
            DatasetRef(path=str(source), format="csv"),
            data_dir,
            max_bytes=10_000,
            max_rows=10,
        )

    assert not (tmp_path / "escape").exists()
    assert not data_dir.exists()
