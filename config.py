from pathlib import Path

# __file__ is the actual pathname of the file that is running
# .resolve(A) converts the path into a full absolute path
# parents is a list of all parent directories
# parents[0] is the immediate parent, which is the folder the file lives in (systematic_trading/)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data"