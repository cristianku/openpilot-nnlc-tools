"""Integration test for Julia training pipeline.

Runs the training script on a small fixture dataset and verifies
it produces a valid model JSON file.
"""

import json
import os
import shutil
import subprocess
import tempfile

import pytest

from nnlc_tools.visualize_model import NNModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "test_data.csv")
RUN_SCRIPT = os.path.join(REPO_ROOT, "training", "run.sh")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is not installed")
def test_training_produces_valid_model():
    """Run training on fixture data and verify JSON output."""
    if not os.path.exists(FIXTURE):
        pytest.skip("Test fixture not found: tests/fixtures/test_data.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy fixture into temp dir as lateral_data.csv (what the training script expects)
        shutil.copy(FIXTURE, os.path.join(tmpdir, "lateral_data.csv"))

        result = subprocess.run(
            ["bash", RUN_SCRIPT, tmpdir, "--cpu"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
        )

        assert result.returncode == 0, (
            f"Training failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

        # Training creates output in a subdirectory named after the CSV (lateral_data/)
        output_dir = os.path.join(tmpdir, "lateral_data")
        assert os.path.isdir(output_dir), (
            f"Output directory not found. Files: {os.listdir(tmpdir)}"
        )

        # Find model JSON output
        model_path = os.path.join(output_dir, "lateral_data.json")
        assert os.path.exists(model_path), (
            f"Model JSON not found. Files in output dir: {os.listdir(output_dir)}"
        )
        with open(model_path) as f:
            model = json.load(f)

        expected_keys = {"input_size", "output_size", "layers"}
        assert expected_keys.issubset(model.keys()), (
            f"Model JSON missing keys. Expected {expected_keys}, got {set(model.keys())}"
        )
        # [nnlc contract] - START
        # Validate the actual serialized model, not only the presence of keys.
        NNModel(model_path).validate_nnlc()
        # [nnlc contract] - END
