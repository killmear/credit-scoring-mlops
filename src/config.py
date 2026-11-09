from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = "default.payment.next.month"
DATA_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/350/" "default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
)
