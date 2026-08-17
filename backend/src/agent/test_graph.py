import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from graph import verify_statement
except (ImportError, ValueError):
    from .graph import verify_statement

verdict = verify_statement("Homoseksualizm jest grzechem")
print(verdict.model_dump_json(indent=2))