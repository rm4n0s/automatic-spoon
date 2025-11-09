import os
import subprocess

from src.db.database import async_close_db, async_init_db


async def test_async_open_close_db():
    db_path = "/tmp/sqlite_test.db"
    await async_init_db(db_path)
    await async_close_db()
    out = subprocess.check_output(["sqlite3", db_path, ".tables"])
    res = [v.strip().decode("utf-8") for v in out.split()]
    expected = [
        "aimodel",
        "aimodelforimage",
        "image",
        "aimodelforengine",
        "engine",
        "job",
    ]
    assert len(expected) == len(res)
    assert all(element in res for element in expected)
    os.remove(db_path)
