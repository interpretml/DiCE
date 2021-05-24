""" Code to automatically run all notebooks as a test.
Adapted from the same code for the Microsoft DoWhy library.
"""

import os
import subprocess
import tempfile

import nbformat
import pytest

NOTEBOOKS_PATH = "docs/source/notebooks/"
notebooks_list = [f.name for f in os.scandir(NOTEBOOKS_PATH) if f.name.endswith(".ipynb")]
# notebooks that should not be run
advanced_notebooks = [
        "DiCE_with_advanced_options.ipynb",  # requires tensorflow 1.x
        "DiCE_getting_started_feasible.ipynb",  # needs changes after latest refactor
        "Benchmarking_different_CF_explanation_methods.ipynb"
        ]

# Adding the dice root folder to the python path so that jupyter notebooks
if 'PYTHONPATH' not in os.environ:
    os.environ['PYTHONPATH'] = os.getcwd()
elif os.getcwd() not in os.environ['PYTHONPATH'].split(os.pathsep):
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + os.pathsep + os.getcwd()


def _check_notebook_cell_outputs(filepath):
    """Convert notebook via nbconvert, collect output and assert if any output cells are not empty.

    :param filepath: file path for the notebook
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook",
                "-y", "--no-prompt",
                "--output", fout.name, filepath]
        subprocess.check_call(args)
        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    for cell in nb.cells:
        if "outputs" in cell:
            if len(cell['outputs']) > 0:
                assert False, "Output cell found in notebook. Please clean your notebook"


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.

    Source of this function: http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

    :param filepath: file path for the notebook
    :returns (parsed nb object, execution errors)
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                # "--ExecutePreprocessor.timeout=600",
                "-y", "--no-prompt",
                "--output", fout.name, filepath]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


# Creating the list of notebooks to run
parameter_list = []
for nb in notebooks_list:
    if nb in advanced_notebooks:
        param = pytest.param(
            nb,
            marks=[pytest.mark.skip, pytest.mark.advanced],
            id=nb)
    else:
        param = pytest.param(nb, id=nb)
    parameter_list.append(param)


@pytest.mark.parametrize("notebook_filename", parameter_list)
def test_notebook(notebook_filename):
    _check_notebook_cell_outputs(NOTEBOOKS_PATH + notebook_filename)
    nb, errors = _notebook_run(NOTEBOOKS_PATH + notebook_filename)
    assert errors == []
