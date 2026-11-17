"""Execute this plain-Python notebook without Jupyter kernel dependencies.

This is a sequential cell smoke check, not a test of Jupyter's kernel protocol.
Usage: python run_notebook.py [notebook.ipynb]
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("01_preprocessing.ipynb")
    path = path.resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    namespace = {"__name__": "__main__"}
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source)
        count += 1
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
        output = buffer.getvalue()
        cell["execution_count"] = count
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": output.splitlines(True)}]
                           if output else [])
        print(f"Cell {count}: OK")
        if output:
            print(output, end="")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved {count} executed cells: {path}")


if __name__ == "__main__":
    main()
