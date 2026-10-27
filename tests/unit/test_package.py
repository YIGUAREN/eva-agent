import eva_agent


def test_version_is_exposed() -> None:
    assert eva_agent.__version__ == "0.1.0"
