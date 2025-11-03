import subprocess
import os
from src.models import init_db, close_db


async def test_open_close_db():
    db_path = "/tmp/sqlite_test.db"
    await init_db(db_path)
    await close_db()
    out = subprocess.check_output(["sqlite3", db_path, ".tables"])
    res = [v.strip().decode("utf-8") for v in out.split()]
    expected = ["aimodel", "engine", "image", "job"]
    assert all(element in res for element in expected)
    os.remove(db_path)

