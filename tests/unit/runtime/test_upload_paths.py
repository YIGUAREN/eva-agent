from pathlib import Path

import pytest

from eva_agent.runtime.uploads import upload_paths, validate_upload_id


def test_upload_paths_are_under_data_dir(tmp_path: Path) -> None:
    paths = upload_paths(tmp_path, "upload_abc123")

    assert paths.root == tmp_path / "uploads" / "upload_abc123"
    assert paths.package == paths.root / "package.bin"
    assert paths.extracted == paths.root / "extracted"


@pytest.mark.parametrize("value", ["", "../x", "a/b", "a\\b", "upload_..", "a" * 65, "upload x"])
def test_validate_upload_id_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        validate_upload_id(value)


def test_validate_upload_id_accepts_generated_shape() -> None:
    assert validate_upload_id("upload_0123abcd") == "upload_0123abcd"
