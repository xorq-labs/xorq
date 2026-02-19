from pathlib import Path

from xorq.catalog.catalog import Catalog


def test_from_name_as_submodule(root_repo):
    name = "my-catalog"
    catalog = Catalog.from_name_as_submodule(root_repo, name, init=True)
    assert isinstance(catalog, Catalog)
    expected_path = Path(root_repo.working_dir) / Catalog.submodule_rel_path / name
    assert catalog.repo_path == expected_path
    submodule_paths = [sm.path for sm in root_repo.submodules]
    assert str(Catalog.submodule_rel_path / name) in submodule_paths


def test_clone_from_as_submodule(root_repo, repo_cloned_bare):
    url = repo_cloned_bare.working_dir
    catalog = Catalog.clone_from_as_submodule(root_repo, url)
    assert isinstance(catalog, Catalog)
    name = Path(url).stem
    expected_path = Path(root_repo.working_dir) / Catalog.submodule_rel_path / name
    assert catalog.repo_path == expected_path
    submodule_paths = [sm.path for sm in root_repo.submodules]
    assert str(Catalog.submodule_rel_path / name) in submodule_paths


def test_mixed_as_submodule(root_repo, repo_cloned_bare):
    name = "my-catalog"
    Catalog.from_name_as_submodule(root_repo, name, init=True)

    url = repo_cloned_bare.working_dir
    Catalog.clone_from_as_submodule(root_repo, url)

    submodule_paths = [sm.path for sm in root_repo.submodules]
    assert str(Catalog.submodule_rel_path / name) in submodule_paths
    assert str(Catalog.submodule_rel_path / Path(url).stem) in submodule_paths
    assert len(submodule_paths) == 2
