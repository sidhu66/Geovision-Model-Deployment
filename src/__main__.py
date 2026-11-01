"""Run the Flask application with: python -m src"""

from __future__ import annotations

import logging

from src.web.app import app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
