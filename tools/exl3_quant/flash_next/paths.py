"""Host paths of the Flash-Next c03 conversion drivers (standard library only).

The drivers ran from the retired ai.tools tree. Their host paths now come from
the environment, defaulting to this repository and /home/m2on/projects/models:

  GUFO_QUANT_OUT       stage receipts and work (/output in the containers);
                       default <repo>/build/flash-next-quant-c03. The c03 run's
                       receipts, which model_job.py's gates read, are in
                       HISTORICAL_OUT.
  GUFO_KYOJIN_RUNTIME  converter runtime from kyojin/fetch_kyojin.sh and
                       build_kyojin.sh (source/ + site/); default
                       <repo>/build/kyojin-runtime.
  GUFO_FN_SOURCE       BF16 Qwen/Qwen3.8-Flash-Next snapshot
                       (de4b8e4d43b917e7706784d8bb445c9af86a3540).
  GUFO_OPS             GPU owner lock and host/NPU checks; default
                       ~/.local/share/gufo/ops (see host_guard.py).
"""
import os
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
HISTORICAL_OUT = Path('/home/m2on/projects/_ai.tools/tmp/test-results/flash-next-quant-20261104')
OUT = Path(os.environ.get('GUFO_QUANT_OUT') or REPO / 'build/flash-next-quant-c03')
RUNTIME = Path(os.environ.get('GUFO_KYOJIN_RUNTIME') or REPO / 'build/kyojin-runtime')
SOURCE = Path(os.environ.get('GUFO_FN_SOURCE')
              or '/run/media/m2on/T5 EVO/AI/model/originals-flash-next-hf-20261030/Qwen-Qwen3.8-Flash-Next')
OPS = Path(os.environ.get('GUFO_OPS') or Path.home() / '.local/share/gufo/ops')
