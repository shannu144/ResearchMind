import pytest_asyncio
from app.database.session import init_db


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """
    Fixture ensuring database tables are initialized before running test cases.
    """
    await init_db()
