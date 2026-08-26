import mlflow
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"

def log_run_info(run_id: str, run_name: str):
    historical_runs = []
    if JSON_FILE_PATH.exists():
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    historical_runs = data
                elif isinstance(data, dict):
                    historical_runs = [data]
        except Exception:
            historical_runs = []

    run_dict = {
        "run_id": run_id,
        "run_name": run_name
    }
    historical_runs.append(run_dict)

    with open(JSON_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(historical_runs, file, indent=4)

def get_metrics_from_runs(tag_name: str, experiment_id: str): 
    searched_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                       filter_string=f"tags.phase = '{tag_name}'",
                       output_format="list")
    
    all_metrics = []
    
    for run in searched_runs:
        metrics_dict = run.data.metrics
        all_metrics.append(metrics_dict)
        
    return all_metrics


