"""``python -m hq.export <bundle_dir>``: validate a written bundle (``hq.export.check``)."""

import sys

from hq.export.check import main

if __name__ == "__main__":
    sys.exit(main())
