"""python3 -I tools/steer COMMAND ...: see cli.py and README.md."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli  # noqa: E402

sys.exit(cli.main())
