"""Allow `python -m arena`."""

import sys

from arena.cli import main

if __name__ == "__main__":
    sys.exit(main())
