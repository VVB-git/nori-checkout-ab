from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "nori.db"


def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")
