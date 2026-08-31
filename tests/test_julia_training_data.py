import os
import shutil
import subprocess

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JULIA_TEST = os.path.join(REPO_ROOT, "tests", "test_training_data.jl")


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is not installed")
def test_julia_training_data_contract():
    result = subprocess.run(
        ["julia", JULIA_TEST],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"Julia training-data tests failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
