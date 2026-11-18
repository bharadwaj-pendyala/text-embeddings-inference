import os
import subprocess
import sys
from pathlib import Path


def test_rocm_flag_imports_on_hpu(tmp_path):
    # flash_attn picks its backend at import time, and is_hpu() only checks that hl-smi runs.
    hl_smi = tmp_path / "hl-smi"
    hl_smi.write_text("#!/bin/sh\nexit 0\n")
    hl_smi.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "USE_FLASH_ATTENTION"}
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from text_embeddings_server.utils.flash_attn import is_hpu, ROCM_HAS_FA_VARLEN\n"
            "assert is_hpu and ROCM_HAS_FA_VARLEN is False",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
