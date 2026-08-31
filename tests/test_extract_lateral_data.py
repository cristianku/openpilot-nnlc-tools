from types import SimpleNamespace

import pytest

from nnlc_tools import logreader
from nnlc_tools.extract_lateral_data import COLUMNS, extract_segment


class FakeMessage(SimpleNamespace):
    def which(self):
        return self.message_type


class FakeLateralControlState(SimpleNamespace):
    def which(self):
        return "torqueState"


def test_extract_segment_converts_logged_torque_to_nnlc_torque(monkeypatch):
    """Catches training on the actuator-sign torque logged by controlsState."""
    car_state = SimpleNamespace(
        vEgo=20.0,
        aEgo=0.0,
        steeringAngleDeg=1.0,
        steeringRateDeg=0.0,
        steeringTorque=0.0,
        steeringPressed=False,
        standstill=False,
    )
    torque_state = SimpleNamespace(
        actualLateralAccel=1.0,
        desiredLateralAccel=1.1,
        output=-0.4,
        saturated=False,
    )
    controls_state = SimpleNamespace(
        lateralControlState=FakeLateralControlState(torqueState=torque_state),
        desiredCurvature=0.002,
        curvature=0.0018,
        active=True,
    )
    messages = [
        FakeMessage(message_type="carState", carState=car_state, logMonoTime=1_000_000_000),
        FakeMessage(message_type="controlsState", controlsState=controls_state, logMonoTime=1_010_000_000),
    ]
    monkeypatch.setattr(logreader, "LogReader", lambda *_args, **_kwargs: messages)

    rows = extract_segment("fake-rlog")

    assert len(rows) == 1
    torque_output = rows[0][COLUMNS.index("torque_output")]
    assert torque_output == pytest.approx(0.4)
