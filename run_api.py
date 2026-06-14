#!/usr/bin/env python3
"""Start the FastAPI backend for the React workbench."""

from api.core.paths import PROJECT_ROOT


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "api"), str(PROJECT_ROOT / "layers")],
    )


if __name__ == "__main__":
    main()
