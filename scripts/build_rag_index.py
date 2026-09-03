"""Index independently vetted reference documents stored as JSONL."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news_guard.config import get_settings
from news_guard.rag import LocalEvidenceStore, index_jsonl_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=get_settings().trusted_documents_path,
        help="JSONL records with id, title, text, url, and publisher fields.",
    )
    args = parser.parse_args()
    settings = get_settings()
    store = LocalEvidenceStore(settings)
    added = index_jsonl_file(store, args.input)
    print(f"Indexed or updated {added} trusted documents. Total in Chroma: {store.count()}")
