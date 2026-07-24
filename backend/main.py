from pathlib import Path

import uvicorn


SRC_DIR = Path(__file__).resolve().parent / "src"


def main() -> None:
    uvicorn.run(
        "api.application:app",
        app_dir=str(SRC_DIR),
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()
