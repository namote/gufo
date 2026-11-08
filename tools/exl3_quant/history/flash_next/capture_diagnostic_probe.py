"""Retry only the failed quarter extraction; retain the completed profile."""
import json
from pathlib import Path
import subprocess
import sys
import time
root=Path('/output/capture-profile-quarter');started=time.monotonic()
result=subprocess.run([sys.executable,'-u','/task/capture_quarter_probe.py'],timeout=1100)
(root/'SUMMARY.json').write_text(json.dumps(dict(status='diagnostics_complete_review_required',
 elapsed_seconds=time.monotonic()-started,stages=[dict(script='capture_quarter_probe.py',exit_code=result.returncode)],
 profile='../capture-profile-quarter-attempt-002/PROFILE.json',quarter='quarter/SUMMARY.json',
 production_changed=False,quality_validated=False),indent=2))
