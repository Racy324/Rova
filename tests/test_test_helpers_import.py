from pathlib import Path


def test_test_helpers_resolve_from_this_repository() -> None:
    import tests
    import tests.tool_helpers as helpers

    assert Path(tests.__file__).resolve().parent == Path(__file__).resolve().parent
    assert Path(helpers.__file__).resolve().parent == Path(__file__).resolve().parent
