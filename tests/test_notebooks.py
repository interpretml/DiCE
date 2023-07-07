""" Code to automatically run all notebooks as a test.
Adapted from the same code for the Microsoft DoWhy library.
"""

import os
import subprocess
import sys
import tempfile

import nbformat
import pytest

NOTEBOOKS_PATH = "docs/source/notebooks/"

# Adding the dice root folder to the python path so that jupyter notebooks
if 'PYTHONPATH' not in os.environ:
    os.environ['PYTHONPATH'] = os.getcwd()
elif os.getcwd() not in os.environ['PYTHONPATH'].split(os.pathsep):
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + os.pathsep + os.getcwd()


def get_notebook_parameter_list():
    notebooks_list = [f.name for f in os.scandir(NOTEBOOKS_PATH) if f.name.endswith(".ipynb")]
    # notebooks that should not be run
    advanced_notebooks = [
            "DiCE_with_advanced_options.ipynb",  # requires tensorflow 1.x
            "DiCE_getting_started_feasible.ipynb",  # needs changes after latest refactor
            "Benchmarking_different_CF_explanation_methods.ipynb"
    ]
    # notebooks that don't need to run on python 3.10
    torch_notebooks_not_3_10 = [
        "DiCE_getting_started.ipynb"
    ]

    # Creating the list of notebooks to run
    parameter_list = []
    for nb in notebooks_list:
        if nb in advanced_notebooks:
            param = pytest.param(
                nb,
                marks=[pytest.mark.skip, pytest.mark.advanced],
                id=nb)
        elif sys.version_info >= (3, 10) and nb in torch_notebooks_not_3_10:
            param = pytest.param(
                nb,
                marks=[pytest.mark.skip, pytest.mark.advanced],
                id=nb)
        else:
            param = pytest.param(nb, id=nb)
        parameter_list.append(param)

    return parameter_list


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
                raise AssertionError("Output cell found in notebook. Please clean your notebook")


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.

    Source of this function: http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

    :param filepath: file path for the notebook
    :returns List of execution errors
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

    return errors


@pytest.mark.parametrize("notebook_filename", get_notebook_parameter_list())
@pytest.mark.notebook_tests()
def test_notebook(notebook_filename):
    _check_notebook_cell_outputs(NOTEBOOKS_PATH + notebook_filename)
    errors = _notebook_run(NOTEBOOKS_PATH + notebook_filename)
    assert len(errors) == 0
