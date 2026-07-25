import os
from pathlib import Path

import uvicorn


SRC_DIR = Path(__file__).resolve().parent / "src"


def _port() -> int:
    value = os.environ.get("PORT", "8000")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    return port


def main() -> None:
    uvicorn.run(
        "api.application:app",
        app_dir=str(SRC_DIR),
        host=os.environ.get("MOOGLE_HOST", "127.0.0.1"),
        port=_port(),
    )


if __name__ == "__main__":
    main()
