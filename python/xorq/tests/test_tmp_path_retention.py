"""The root conftest's ``tmp_path`` retention, run in an inner pytest session."""

import re
import stat
import subprocess
import sys
import textwrap

import pytest


MIXED_OUTCOMES = textwrap.dedent(
    """
    from pathlib import Path

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
        # as git leaves its objects: Windows refuses to remove a read-only file
        (read_only / "removed").write_text("")
        (read_only / "removed").chmod(0o444)
        read_only.chmod(0o555)
        outside = Path(__file__).parent / "outside"
        outside.write_text("")
        outside.chmod(0o444)
        (tmp_path / "link").symlink_to(outside)
        for mode in (0o000, 0o300):
            unreadable = tmp_path / f"unreadable{mode:o}"
            (unreadable / "nested").mkdir(parents=True)
            (unreadable / "nested" / "removed").write_text("")
            unreadable.chmod(mode)


    @pytest.mark.xfail(raises=AssertionError, strict=True)
    def test_xfails(tmp_path):
        (tmp_path / "kept").write_text("")
        assert False


    def test_skips_in_call(tmp_path):
        (tmp_path / "kept").write_text("")
        pytest.skip("in call")


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


XDIST = pytest.mark.parametrize("xdist", ((), ("-n", "2")), ids=("in-process", "xdist"))


@pytest.fixture(autouse=True)
def temproot(tmp_path, monkeypatch):
    # where an inner run without --basetemp numbers its dirs, as pytest does
    # under the system temp dir on a developer's machine
    path = tmp_path / "temproot"
    path.mkdir()
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(path))
    return path


def run_inner(tmp_path, source, *args, policy="failed", basetemp=True):
    """Run source as a test file; return the --basetemp given and the result."""
    test_file = tmp_path / "test_inner.py"
    test_file.write_text(source)
    given = (f"--basetemp={tmp_path / 'basetemp'}",) if basetemp else ()
    result = subprocess.run(
        (
            *(sys.executable, "-m", "pytest", "-p", "xorq.conftest"),
            *("-p", "no:cacheprovider", "-p", "no:randomly"),
            *("-o", f"tmp_path_retention_policy={policy}"),
            *given,
            *args,
            *(f"--rootdir={tmp_path}", str(test_file)),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    return tmp_path / "basetemp", result


def numbered_basetemps(temproot):
    return tuple(temproot.glob("pytest-of-*/pytest-[0-9]*"))


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
        "test_xfails0/kept",
        "test_skips_in_call0/kept",
        "test_setup_error0/kept",
        "test_teardown_error0/kept",
        "test_last_before_session_teard0/kept",
    }
    # emptied, not removed, so the two names truncated alike stay apart
    assert {path.name for path in basetemp.glob("test_parametrized_*")} >= {
        "test_parametrized_with_a_name_0",
        "test_parametrized_with_a_name_1",
    }
    assert "tmp_path cleanup errors" not in result.stdout
    # cleanup removed the link and left what it pointed to alone
    assert stat.S_IMODE((tmp_path / "outside").stat().st_mode) == 0o444


GREEN_WITH_READ_ONLY_SESSION_DIR = textwrap.dedent(
    """
    import pytest


    @pytest.fixture(scope="session")
    def annex(tmp_path_factory):
        # as git-annex leaves its object dirs: pytest's own removal skips these
        objects = tmp_path_factory.mktemp("annex") / "objects"
        objects.mkdir()
        (objects / "removed").write_text("")
        objects.chmod(0o555)
        return objects


    @pytest.mark.parametrize("i", range(4))
    def test_passes(annex, tmp_path, i):
        (tmp_path / "removed").write_text("")
    """
)


@XDIST
def test_failed_policy_removes_basetemp_after_a_green_run(tmp_path, temproot, xdist):
    _, result = run_inner(
        tmp_path, GREEN_WITH_READ_ONLY_SESSION_DIR, *xdist, basetemp=False
    )
    assert result.returncode == 0, result.stdout
    assert not numbered_basetemps(temproot)
    assert "tmp_path cleanup errors" not in result.stdout


@XDIST
def test_failed_policy_keeps_a_given_basetemp(tmp_path, xdist):
    basetemp, result = run_inner(
        tmp_path, "def test_passes(tmp_path):\n    pass\n", *xdist
    )
    assert result.returncode == 0, result.stdout
    assert basetemp.is_dir()


CLEANUP_ERROR = textwrap.dedent(
    """
    import pytest

    import xorq.conftest


    def refuse(path):
        raise PermissionError(f"cannot chmod {path}")


    # as for a dir another user owns; a conftest, so an xdist controller
    # patches it too, and the hooks look the name up at cleanup
    xorq.conftest._make_removable = refuse


    @pytest.fixture
    def teardown_error(tmp_path):
        yield
        raise RuntimeError("fixture teardown")
    """
)


@XDIST
def test_cleanup_error_is_reported_not_failed(tmp_path, xdist):
    (tmp_path / "conftest.py").write_text(CLEANUP_ERROR)
    _, result = run_inner(
        tmp_path,
        "def test_passes(tmp_path):\n    pass\n",
        *xdist,
        basetemp=False,
    )
    assert result.returncode == 0, result.stdout
    summary = result.stdout.split("tmp_path cleanup errors", 1)[1]
    # the test's dir, then the session's: the controller's under xdist
    assert re.findall(r"cannot chmod \S+/(\S+)'", summary) == [
        "test_passes0",
        "pytest-0",
    ], result.stdout


def test_cleanup_error_keeps_the_teardown_error(tmp_path):
    (tmp_path / "conftest.py").write_text(CLEANUP_ERROR)
    _, result = run_inner(
        tmp_path,
        "def test_teardown(teardown_error):\n    pass\n",
        policy="none",
    )
    assert result.returncode == 1, result.stdout
    assert "fixture teardown" in result.stdout
    assert "tmp_path cleanup errors" in result.stdout
