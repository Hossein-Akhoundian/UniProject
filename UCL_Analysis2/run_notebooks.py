"""Execute all UCL notebooks in order and store their outputs in-place."""

import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
NOTEBOOK_DIR = HERE / "Notebooks"


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        print(f"Executing {path.name} ...", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        )
        client.execute()
        nbformat.write(notebook, path)
        print(f"Completed {path.name}", flush=True)


if __name__ == "__main__":
    main()
