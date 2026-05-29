from __future__ import annotations
import os
import sys
from code.main_all import main

if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
