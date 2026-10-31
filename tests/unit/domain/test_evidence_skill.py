from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.run import InputKind

_DIGEST = "0" * 64


def bundle(*paths: str) -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.UPLOAD,
        root="/tmp/evidence",
        files=[EvidenceFile(path=path, size=1, sha256=_DIGEST) for path in paths],
        manifest_sha256=_DIGEST,
    )


def test_root_skill_md_is_a_skill_package() -> None:
    assert bundle("SKILL.md", "scripts/run.py").has_skill_manifest() is True


def test_root_skill_yaml_variants_are_skill_packages() -> None:
    assert bundle("skill.yaml").has_skill_manifest() is True
    assert bundle("skill.yml").has_skill_manifest() is True


def test_nested_manifest_is_a_skill_package() -> None:
    # A packaged skill commonly keeps its manifest under a wrapper directory;
    # detection must not depend on the evidence root.
    assert bundle("examples/SKILL.md", "qa.txt").has_skill_manifest() is True


def test_wrapped_package_with_root_qa_is_a_skill_package() -> None:
    assert (
        bundle("ssh-collector/SKILL.md", "ssh-collector/scripts/run.py", "qa.jsonl").has_skill_manifest()
        is True
    )


def test_manifests_under_two_top_level_directories_are_a_skill_package() -> None:
    # One wrapper directory is not a requirement; every manifest-bearing
    # directory counts as a package.
    assert bundle("a/x/SKILL.md", "b/y/SKILL.md").has_skill_manifest() is True


def test_plain_upload_is_not_a_skill_package() -> None:
    assert bundle("qa.jsonl").has_skill_manifest() is False


def test_upload_without_any_manifest_is_not_a_skill_package() -> None:
    assert bundle("scripts/run.py", "docs/readme.md", "qa.jsonl").has_skill_manifest() is False
