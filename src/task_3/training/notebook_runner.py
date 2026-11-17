from pathlib import Path
import sys
import traceback

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

class NotebookShell(InteractiveShell):
    def enable_gui(self, gui=None):
        if gui not in {None, "inline"}:
            raise ValueError("This runner supports only inline figures.")

def execute(path):
    path = Path(path)
    notebook = nbformat.read(path, as_version=4)
    shell = NotebookShell.instance()
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        outputs = []
        for name, text in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
            if text:
                outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
        for output in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        error = result.error_before_exec or result.error_in_exec
        if error:
            outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error), traceback=traceback.format_exception(error)))
        cell.outputs, cell.execution_count = outputs, result.execution_count
        nbformat.write(notebook, path)
        if error:
            raise RuntimeError(f"{path.name}, cell{i}: {error}") from error
        print(f"  cell{i}: completed", flush=True)

if __name__ == "__main__":
    execute(sys.argv[1])
