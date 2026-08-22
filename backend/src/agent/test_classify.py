import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from tools import classify_support, retrieve
except (ImportError, ValueError):
    from .tools import classify_support, retrieve

statement = "Antykoncepcja jest grzechem"
chunks = retrieve(statement, top_k=5)
verdict = classify_support(statement, chunks)

print(verdict.model_dump_json(indent=2))
