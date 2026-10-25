"""Local checks; never connects to hardware or controls real services."""

import os
import pathlib
import subprocess
import tempfile
from unittest.mock import Mock, patch

import sweep_acceptance as sweep


def main():
    baseline = sweep.parse_args(["--spec-type", "none", "--temperature", "0"])
    with patch.dict(os.environ, {"DFLASH_XBOX_RPC": "unexpected", "DFLASH_SUDO_PASSWORD": "test"}):
        cmd, env = sweep.server_command(baseline, 0, 0)
    assert not any(x.startswith("--spec-draft") for x in cmd)
    assert "DFLASH_XBOX_RPC" not in env and "DFLASH_SUDO_PASSWORD" not in env
    assert sweep.configurations(baseline) == [(0, 0, 0, "baseline")]
    custom = sweep.parse_args(["--n-max", "5", "--p-min", "0.3", "--temperature", "1", "--draft-model", "q2.gguf"])
    assert sweep.configurations(custom) == [(5, 0.3, 1, "single")]
    assert "q2.gguf" in sweep.server_command(custom, 5, 0.3)[0]

    proc = Mock()
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("server", 5), 0]
    sweep.CURRENT_PROC = proc
    sweep.stop_process()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2 and sweep.CURRENT_PROC is None

    with tempfile.TemporaryDirectory() as directory:
        argv = ["--spec-type", "none", "--output", str(pathlib.Path(directory) / "results.jsonl")]
        for active in (False, True):
            with patch.object(sweep, "service_active", return_value=active), \
                 patch.object(sweep, "service_command") as service, \
                 patch.object(sweep, "wait_for_health", return_value=True), \
                 patch.object(sweep.signal, "signal"), \
                 patch.object(sweep, "run_single_benchmark", side_effect=RuntimeError("benchmark failed")):
                try:
                    sweep.main(argv)
                except RuntimeError as error:
                    assert str(error) == "benchmark failed"
                else:
                    raise AssertionError("benchmark failure must propagate")
                assert [call.args[0] for call in service.call_args_list] == (["stop", "start"] if active else [])
        with patch.object(sweep, "service_active", return_value=True), \
             patch.object(sweep, "service_command"), \
             patch.object(sweep, "wait_for_health", return_value=False), \
             patch.object(sweep.signal, "signal"), \
             patch.object(sweep, "configurations", return_value=[]):
            try:
                sweep.main(argv)
            except RuntimeError as error:
                assert "did not recover" in str(error)
            else:
                raise AssertionError("restoration failure must propagate")
    print("sweep checks passed")


if __name__ == "__main__":
    main()
