from pathlib import Path
from dotenv import load_dotenv

# Load .env from the pa package directory so `pa run` works from any cwd
_dotenv = Path(__file__).parent.parent / ".env"
load_dotenv(_dotenv, override=True)

from pa.cli import app  # noqa: E402  (dotenv must load before pa imports)

app()
