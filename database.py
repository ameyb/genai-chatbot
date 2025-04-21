import os
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("PG_VECTOR_HOST")
user = os.getenv("PG_VECTOR_USER")
password = os.getenv("PG_VECTOR_PASSWORD")
COLLECTION_NAME = os.getenv("PGDATABASE")
CONNECTION_STRING = "postgresql+psycopg://yugabyte:@127.0.0.1:5433/yugabyte"
#CONNECTION_STRING = f"postgresql+psycopg://{user}:{password}@{host}:6042/{COLLECTION_NAME}"

