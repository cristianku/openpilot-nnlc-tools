"""Future Event variants must not discard the remaining training samples."""

import bz2
import struct

import capnp
import pytest
import zstandard

from nnlc_tools import logreader
from nnlc_tools.extract_lateral_data import COLUMNS, extract_segment


def unknown_event_bytes(timestamp):
    event = logreader.capnp_log.Event.new_message(logMonoTime=timestamp)
    event.init("carState")
    raw = bytearray(event.to_bytes())
    # These small fixtures have one segment and a direct root struct pointer.
    assert struct.unpack_from("<I", raw)[0] == 0
    root_pointer = struct.unpack_from("<Q", raw, 8)[0]
    assert root_pointer & 3 == 0
    root_start = 16 + ((root_pointer >> 2) & 0x3FFFFFFF) * 8
    schema = logreader.capnp_log.Event.schema.node.struct
    struct.pack_into("<H", raw, root_start + schema.discriminantOffset * 2,
                     schema.discriminantCount + 1)
    return bytes(raw)


def training_events():
    car = logreader.capnp_log.Event.new_message(logMonoTime=1_000_000_000)
    car.init("carState").vEgo = 20.0
    events = [car.to_bytes()]
    for timestamp, output in [(1_010_000_000, -0.4), (1_040_000_000, -0.6)]:
        event = logreader.capnp_log.Event.new_message(logMonoTime=timestamp)
        controls = event.init("controlsState")
        controls.desiredCurvature = 0.002
        controls.curvature = 0.0018
        torque = controls.lateralControlState.init("torqueState")
        torque.actualLateralAccel = 1.0
        torque.desiredLateralAccel = 1.1
        torque.output = output
        events.append(event.to_bytes())
    return events


@pytest.mark.parametrize("compression", ["raw", "zst", "bz2"])
def test_extract_preserves_rows_after_future_events(tmp_path, capsys, compression):
    car, first, last = training_events()
    # Two unsupported events between two valid training samples.
    raw = car + first + unknown_event_bytes(1_020_000_000)
    raw += unknown_event_bytes(1_030_000_000) + last
    if compression == "zst":
        raw = zstandard.ZstdCompressor().compress(raw)
    elif compression == "bz2":
        raw = bz2.compress(raw)
    path = tmp_path / f"rlog.{compression}"
    path.write_bytes(raw)

    rows = extract_segment(str(path))

    assert len(rows) == 2
    assert [row[COLUMNS.index("timestamp")] for row in rows] == [1.01, 1.04]
    assert [row[COLUMNS.index("torque_output")] for row in rows] == pytest.approx([0.4, 0.6])
    output = capsys.readouterr().out
    assert "Skipped 2 unknown event(s)" in output
    assert str(path) in output
    assert "Error processing" not in output


@pytest.mark.parametrize("error", [RuntimeError("broken reader"), capnp.KjException("invalid pointer")])
def test_extract_reports_other_message_type_errors(monkeypatch, capsys, error):
    class BrokenMessage:
        def which(self):
            raise error

    monkeypatch.setattr(logreader, "LogReader", lambda *_args, **_kwargs: [BrokenMessage()])

    assert extract_segment("broken-rlog") == []

    output = capsys.readouterr().out
    assert "Error processing broken-rlog" in output
    assert str(error) in output
    assert "Skipped" not in output
