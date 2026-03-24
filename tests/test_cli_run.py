import os
import subprocess
import sys


def test_cli_run_awgn_smoke() -> None:
    cmd = [
        sys.executable,
        "-m",
        "comm_ai.experiments.run_experiment",
        "--config",
        "src/comm_ai/config/experiments/awgn_smoke.yaml",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH','')}"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "Saved run to:" in result.stdout
