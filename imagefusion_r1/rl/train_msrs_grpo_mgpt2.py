#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_IMPORT_STARTED = time.time()
print(
    "[ENTRY] importing MSRS GRPO trainer "
    f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
    f"local_rank={os.environ.get('LOCAL_RANK', '?')}",
    flush=True,
)
from imagefusion_r1.rl.grpo_trainer_mgpt2 import main
print(
    "[ENTRY] MSRS GRPO trainer import complete "
    f"elapsed_sec={time.time() - _IMPORT_STARTED:.2f}",
    flush=True,
)

#入口
if __name__ == "__main__":
    main()
