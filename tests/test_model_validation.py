# [nnlc contract] - START
"""Reject models that Sunnypilot would consume with wrong inputs or direction."""
import json
import sys

import numpy as np
import pytest

from nnlc_tools import visualize_model


RUNTIME_INPUTS = [
    "v_ego", "actual_lateral_accel", "lateral_jerk", "roll",
    "actual_lateral_accel_tm03", "actual_lateral_accel_tm02", "actual_lateral_accel_tm01",
    "actual_lateral_accel_tp03", "actual_lateral_accel_tp06", "actual_lateral_accel_tp10",
    "actual_lateral_accel_tp15", "roll_tm03", "roll_tm02", "roll_tm01", "roll_tp03",
    "roll_tp06", "roll_tp10", "roll_tp15",
]


def model_params(slope=0.3):
    weights = [0.0] * 18
    weights[1] = slope
    return {
        "input_size": 18,
        "output_size": 1,
        "input_vars": RUNTIME_INPUTS.copy(),
        "input_mean": [[0.0] for _ in range(18)],
        "input_std": [[1.0] for _ in range(18)],
        "layers": [{"dense_1_W": [weights], "dense_1_b": [[0.0]], "activation": "identity"}],
    }


def run_validation_cli(tmp_path, monkeypatch, params):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(params))
    data_path = tmp_path / "data.csv"
    data_path.write_text("v_ego,actual_lateral_accel,torque_output\n20,1,0.3\n")
    plots = []
    monkeypatch.setattr(visualize_model, "plot_lat_accel_vs_torque", lambda *_: plots.append("lat"))
    monkeypatch.setattr(visualize_model, "plot_torque_vs_speed", lambda *_: plots.append("speed"))
    monkeypatch.setattr(sys, "argv", ["nnlc-validate", str(model_path), str(data_path),
                                     "-o", str(tmp_path / "plots")])
    return plots


@pytest.mark.parametrize("defect", ["legacy_order", "negative_torque", "zero_std", "unsupported_activation"])
def test_validation_cli_rejects_incompatible_model_before_plotting(tmp_path, monkeypatch, capsys, defect):
    params = model_params()
    if defect == "legacy_order":
        params["input_vars"].append(params["input_vars"].pop(2))
    elif defect == "negative_torque":
        params = model_params(slope=-0.3)
    elif defect == "zero_std":
        params["input_std"][2] = [0.0]
    else:
        params["layers"][0]["activation"] = "tanh"
    plots = run_validation_cli(tmp_path, monkeypatch, params)
    with pytest.raises(SystemExit) as exc:
        visualize_model.main()
    assert exc.value.code == 1
    assert "ERROR:" in capsys.readouterr().out
    assert plots == []
    assert not (tmp_path / "plots").exists()


def test_validation_cli_accepts_runtime_compatible_positive_torque_model(tmp_path, monkeypatch, capsys):
    plots = run_validation_cli(tmp_path, monkeypatch, model_params())
    visualize_model.main()
    assert plots == ["lat", "speed"]
    assert "NNLC compatibility validation passed" in capsys.readouterr().out


def test_sigmoid_matches_sunnypilot_safe_exp_limit():
    # Sunnypilot clamps the exponent to 11; the validator must evaluate the
    # deployed network with that same activation, including its negative tail.
    values = np.array([-20.0, -2.0, 0.0, 2.0, 20.0], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(np.minimum(-values, 11.0)))
    np.testing.assert_allclose(visualize_model.sigmoid(values), expected, rtol=1e-6)
# [nnlc contract] - END
