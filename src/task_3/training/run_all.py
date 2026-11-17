from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.training.experiment import STUDIES, OUTPUT, prepare_run, read_json, final_report
from src.task_3.training.evaluation import now, write_json

def execute(path):
    print(f"Running {path.relative_to(ROOT)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src/task_3/training/notebook_runner.py"), str(path)], cwd=ROOT, check=True)
    print(f"Completed {path.name}", flush=True)

def main():
    kernel_root = Path("/tmp/task3-improved-jupyter")
    for variable, name in [("JUPYTER_RUNTIME_DIR", "runtime"), ("IPYTHONDIR", "ipython"), ("JUPYTER_CONFIG_DIR", "config")]:
        directory = kernel_root / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    kernel = kernel_root / "kernels/task3-improved"
    kernel.mkdir(parents=True, exist_ok=True)
    write_json(kernel / "kernel.json", {"argv": [str(ROOT / "nlp_venv/bin/python"), "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Task 3 improved", "language": "python"})
    os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + os.environ["JUPYTER_PATH"] if os.environ.get("JUPYTER_PATH") else "")
    expanded_test = (OUTPUT / "all_conditions_test_v1/previous_scope/manifest.json").exists()
    if expanded_test:
        from src.task_3.training.all_tests import prepare
        prepare()
    else:
        prepare_run()
    for level, models in STUDIES.items():
        for model_id, name in models.items():
            execute(ROOT / f"src/task_3/training/level_{level}/{name}.ipynb")
    frozen = read_json(OUTPUT / "final_test/frozen_config.json")
    selected = frozen["selected_level3_model"]
    if not expanded_test:
        for level, model_id in [(1, "L1-a"), (1, "L1-b"), (1, "L1-c"), (1, "L1-e"), (3, selected)]:
            execute(ROOT / f"src/task_3/training/level_{level}/{STUDIES[level][model_id]}.ipynb")
    final_report()
    directory = OUTPUT / "all_conditions_test_v1" if expanded_test else OUTPUT
    config_path = directory / "frozen_config.json" if expanded_test else OUTPUT / "final_test/frozen_config.json"
    write_json(directory / "execution_complete.json", {"completed_at_wib": now(), "validation_notebooks": sum(map(len, STUDIES.values())), "validation_selected_nlp_model": selected,
               "all_conditions_tested": expanded_test, "test_configuration_sha256": __import__("src.task_3.training.testing", fromlist=["sha256"]).sha256(config_path)})
    print("Completed all levels and the requested final-test comparisons.", flush=True)

if __name__ == "__main__":
    main()
