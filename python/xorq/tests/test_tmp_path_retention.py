"""The root conftest's ``tmp_path`` retention, run in an inner pytest session."""

import subprocess
import sys
import textwrap


MIXED_OUTCOMES = textwrap.dedent(
    """
    import pytest


    @pytest.fixture(scope="session")
    def session_teardown_error():
        yield
        raise RuntimeError("session teardown")


    @pytest.fixture
    def setup_error(tmp_path):
        (tmp_path / "kept").write_text("")
        raise RuntimeError("setup")


    @pytest.fixture
    def teardown_error(tmp_path):
        (tmp_path / "kept").write_text("")
        yield
        raise RuntimeError("teardown")


    def test_passes(tmp_path):
        read_only = tmp_path / "objects"
        read_only.mkdir()
        (read_only / "removed").write_text("")
        read_only.chmod(0o555)


    def test_fails(tmp_path):
        (tmp_path / "kept").write_text("")
        assert False


    def test_setup_error(setup_error):
        pass


    def test_teardown_error(teardown_error):
        pass


    @pytest.mark.parametrize("i", range(2))
    def test_parametrized_with_a_name_long_enough_to_truncate(tmp_path, i):
        (tmp_path / "removed").write_text("")


    def test_last_before_session_teardown(tmp_path, session_teardown_error):
        (tmp_path / "kept").write_text("")
    """
)


def run_inner(tmp_path, source):
    test_file = tmp_path / "test_inner.py"
    test_file.write_text(source)
    basetemp = tmp_path / "basetemp"
    result = subprocess.run(
        (
            *(sys.executable, "-m", "pytest", "-p", "xorq.conftest"),
            *("-p", "no:cacheprovider", "-p", "no:randomly"),
            *("-o", "tmp_path_retention_policy=failed"),
            *(f"--basetemp={basetemp}", f"--rootdir={tmp_path}", str(test_file)),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    return basetemp, result


def test_failed_policy_keeps_failed_and_errored_tests_files(tmp_path):
    basetemp, result = run_inner(tmp_path, MIXED_OUTCOMES)
    assert result.returncode == 1, result.stdout
    # pytest also links each name's latest dir as <name>current
    assert {
        str(path.relative_to(basetemp))
        for path in basetemp.glob("*/*")
        if not path.parent.is_symlink()
    } == {
        "test_fails0/kept",
        "test_setup_error0/kept",
        "test_teardown_error0/kept",
        "test_last_before_session_teard0/kept",
    }
    # emptied, not removed, so the two names truncated alike stay apart
    assert {path.name for path in basetemp.glob("test_parametrized_*")} >= {
        "test_parametrized_with_a_name_0",
        "test_parametrized_with_a_name_1",
    }


def test_failed_policy_removes_basetemp_after_a_green_run(tmp_path):
    basetemp, result = run_inner(tmp_path, "def test_passes(tmp_path):\n    pass\n")
    assert result.returncode == 0, result.stdout
    assert not basetemp.exists()
