import pytest
import tempfile
import shutil
import os

file_list_lax = ["vk_ref.ipynb", "vk_count.ipynb", "vk_sim.ipynb"]  #* modify this list
file_list_strict = []  #* modify this list
# make sure to reload and execute notebook after making changes for them to take effect in pytest
# add '# NBVAL_IGNORE_OUTPUT' (without quotes) at the tops of cells I don't expect to be the same when runing --nbval (strict)
# add '# NBVAL_CHECK_OUTPUT' (without quotes) at the tops of cells I expect to be the same when runing --nbval-lax (lax)

@pytest.fixture(scope="session")
def temp_working_dir():
    """Create a temporary working directory and clean up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir  # Provide the temp dir to the test
    shutil.rmtree(temp_dir)  # Cleanup after all tests

@pytest.mark.parametrize("file_name", file_list_lax)
def test_run_lax(temp_working_dir, file_name):    
    """Run all Jupyter notebooks in a temporary directory using nbval."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebook_path = os.path.join(base_dir, file_name)

    result = pytest.main(["--nbval-lax", "--tb=short", notebook_path], plugins=[])
    assert result == 0, "Notebook execution failed"

@pytest.mark.parametrize("file_name", file_list_strict)
def test_run_identical_strict(temp_working_dir, file_name):
    """Run all Jupyter notebooks in a temporary directory using nbval."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebook_path = os.path.join(base_dir, file_name)

    result = pytest.main(["--nbval", "--tb=short", notebook_path], plugins=[])
    assert result == 0, "Notebook execution failed"