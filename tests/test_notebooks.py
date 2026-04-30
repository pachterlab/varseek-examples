import pytest
import shutil
import os
import nbformat

#!!! currently these tests don't seem to work - to test, just copy the notebook into a temp directory manually (e.g., trash) and run

file_list_lax = ["vk_ref.ipynb", "vk_count.ipynb", "vk_sim.ipynb"]  #* modify this list
file_list_strict = []  #* modify this list
# make sure to reload and execute notebook after making changes for them to take effect in pytest
# to test each notebook individually, since the debug button doesn't work with this Jupyter debugging: copy the notebook into a temp directory manually (e.g., trash) and run
# add '# NBVAL_IGNORE_OUTPUT' (without quotes) at the tops of cells I don't expect to be the same when runing --nbval (strict)
# add '# NBVAL_CHECK_OUTPUT' (without quotes) at the tops of cells I expect to be the same when runing --nbval-lax (lax)

# # good if I want to keep a tmp directory for all tests, but not if I want a new one for each test
# @pytest.fixture(scope="session")
# def temp_working_dir():
#     """Create a temporary working directory and clean up after the test."""
#     temp_dir = tempfile.mkdtemp()
#     yield temp_dir  # Provide the temp dir to the test
#     shutil.rmtree(temp_dir)  # Cleanup after all tests

def clear_notebook_output(notebook_path):
    """Remove all outputs from a Jupyter notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []  # Clear output for code cells
        if "execution_count" in cell:
            cell["execution_count"] = None  # Reset execution count

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

@pytest.mark.parametrize("file_name", file_list_lax)
def test_run_lax(tmp_path, file_name):    
    """Run all Jupyter notebooks in a temporary directory using nbval."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebook_path = os.path.join(base_dir, file_name)

    temp_notebook_path = os.path.join(tmp_path, file_name)
    shutil.copy(notebook_path, temp_notebook_path)
    clear_notebook_output(temp_notebook_path)  # avoid Unrun reference cell has outputs

    print(f"Testing notebook {temp_notebook_path}")
    result = pytest.main(["--nbval-lax", "--tb=short", "-vv", temp_notebook_path], plugins=[])
    assert result == 0, "Notebook execution failed"

@pytest.mark.parametrize("file_name", file_list_strict)
def test_run_identical_strict(tmp_path, file_name):
    """Run all Jupyter notebooks in a temporary directory using nbval."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebook_path = os.path.join(base_dir, file_name)

    temp_notebook_path = os.path.join(tmp_path, file_name)
    shutil.copy(notebook_path, temp_notebook_path)
    # clear_notebook_output(temp_notebook_path)  # avoid Unrun reference cell has outputs

    print(f"Testing notebook {temp_notebook_path}")
    result = pytest.main(["--nbval", "--tb=short", "-vv", temp_notebook_path], plugins=[])
    assert result == 0, "Notebook execution failed"
