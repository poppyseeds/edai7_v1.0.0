"""Local entry point for the API."""

from __future__ import annotations

import argparse

import uvicorn

from app.utils.data_utils import create_sample_churn_dataset
from app.utils.logging_config import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Synthetic-AI platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--prepare-sample",
        action="store_true",
        help="Write datasets/sample/customer_churn.csv and exit.",
    )
    args = parser.parse_args()
    if args.prepare_sample:
        path = create_sample_churn_dataset("datasets/sample/customer_churn.csv")
        print(f"Wrote {path}")
        return
    print("API:     uvicorn app.api.main:app --reload")
    print("UI:      streamlit run frontend/dashboard.py")
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
