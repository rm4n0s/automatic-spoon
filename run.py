# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

import multiprocessing as mp

from src.main import main

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
