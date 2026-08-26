from src.configs.config import params_config
from src.data.generate_eval_dataset import generate_evaluation_dataset
from src.evals.application_evals.evaluate_rag_app import evaluate_app
from dotenv import load_dotenv
from pathlib import Path
from typing import Literal
from langfuse import get_client
import mlflow
import dagshub
import json
import logging
from mlflow_utils import log_run_info

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent

# to compare the params we have to remove nesting 
def flatten_params(param_dict: dict):
    config_dict = {}

    for key, value in param_dict.items():
        if isinstance(value, dict):
            config_dict.update(flatten_params(value))
        else:
            if key in config_dict:
                raise ValueError(f"duplicate key found: {key}")
            config_dict[key] = value
        
    return config_dict

# get the latest evaluation results or reports
def get_latest_results(path: Path, pattern: str = "*.json") -> Path:
    files = list(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {path}")
    
    recent_file = max(files, key=lambda f: f.stat().st_mtime)
    return recent_file

# extract the metrics from the result json file
def get_metrics_from_results(result_json: str | Path) -> dict[str, float]:
    with open(result_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    test_cases = data.get("testCases", [])
    metric_scores: dict[str, list[float]] = {}

    for tc in test_cases:
        for metric_data in tc.get("metricsData", []):
            raw_name = metric_data.get("name", "")
            metric_name = raw_name.replace("[GEval]", "").strip()
            score = metric_data.get("score")
            if score is not None:
                metric_scores.setdefault(metric_name, []).append(float(score))

    metrics: dict[str, float] = {}
    for name, scores in metric_scores.items():
        if scores:
            metrics[name] = round(sum(scores) / len(scores), 2)

    return metrics


def import_system_prompt(label: str = "staging") -> str:
    langfuse = get_client()
    prompt_obj = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=label
    )
    return prompt_obj.prompt

def get_artifact_name(artifact_type: Literal["eval_dataset", "golden_dataset"], save_artifact_dir: str) -> tuple[str, str]:
    DATA_PATH = ROOT_DIR / "data" / "evaluation"
    
    if artifact_type == "eval_dataset":
        artifact_path = (DATA_PATH / "eval_dataset" / params_config.evaluation_dataset.evaluation_dataset_filename).with_suffix(".json")
        return (artifact_path.as_posix(), save_artifact_dir)
    
    elif artifact_type == "golden_dataset":
        artifact_path = (DATA_PATH / "goldens" / params_config.golden_dataset.golden_dataset_filename).with_suffix(".json")
        return (artifact_path.as_posix(), save_artifact_dir)
    
    raise ValueError(f"Unknown artifact_type: {artifact_type}")
    
def return_code_files() -> list[str]:
    CODE_PATHS = ROOT_DIR / "src"
    
    code_files = [
        CODE_PATHS / "app" / "rag_workflow.py",
        CODE_PATHS / "app" / "load_system_prompt.py",
        CODE_PATHS / "app" / "system_prompt_versioning.py",
        CODE_PATHS / "configs" / "config.py",
        CODE_PATHS / "configs" / "config_types.py",
        CODE_PATHS / "data" / "generate_eval_dataset.py",
        CODE_PATHS / "data" / "generate_goldens.py",
        CODE_PATHS / "evals" / "application_evals" / "evaluate_rag_app.py",
        ROOT_DIR / "mlflow_utils.py",
        ROOT_DIR / "execute_experiment_pipeline.py",
        ROOT_DIR / "main.py",
        ROOT_DIR / "params.yaml",
    ]
    
    return [file.as_posix() for file in code_files if file.exists()]



if __name__ == "__main__":
    # initialize dagshub and mlflow
    dagshub.init(repo_owner='hariom-mishra', repo_name='learn-llmops-rag', mlflow=True)
    
    # set the tracking server
    mlflow.set_tracking_uri("https://dagshub.com/hariom-mishra/learn-llmops-rag.mlflow")
    
    # set the experiment name
    mlflow.set_experiment("rag-app")
    
    # do the logging
    logger = logging.getLogger(name="MLflow logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt=formatter)
        logger.addHandler(handler)
    
    with mlflow.start_run() as run:
    
        # get all the params
        all_params = params_config.model_dump()
        params_dict = flatten_params(all_params)
        
        # log params on mlflow
        mlflow.log_params(params_dict)
        logger.info("Parameters Logged")
        
        # generate evaluation data
        generate_evaluation_dataset()
        logger.info("evaluation dataset created")
        
        # run the eval pipeline
        evaluate_app()
        logger.info("evaluation complete")
        
        # paths for latest reports
        RESULT_DIR = ROOT_DIR / "reports" / params_config.evaluation.display_config.result_dir
        REPORT_DIR = ROOT_DIR / "reports" / params_config.evaluation.display_config.report_dir
        
        # get the latest files after eval pipeline
        results_file_path = get_latest_results(RESULT_DIR, "*.json")
        report_file_path = get_latest_results(REPORT_DIR, "*.md")

        results_file = results_file_path.as_posix()
        report_file = report_file_path.as_posix()
        
        # log the evaluation results
        mlflow.log_artifact(results_file, "results")
        mlflow.log_artifact(report_file, "reports")
        logger.info("Results and Report logged")
        
        # log the metrics
        metrics = get_metrics_from_results(results_file)
        mlflow.log_metrics(metrics)
        logger.info("Metrics Logged")
        
        # log the system prompt
        system_prompt = import_system_prompt(params_config.rag_app.prompt_label)
        mlflow.log_text(system_prompt,
                        artifact_file="system_prompt.txt")
        logger.info("System prompt logged")
        
        # log the datasets
        eval_dataset_artifact = get_artifact_name(artifact_type="eval_dataset",
                                                  save_artifact_dir="eval_dataset")
        golden_dataset_artifact = get_artifact_name(artifact_type="golden_dataset",
                                                    save_artifact_dir="golden_dataset")
        mlflow.log_artifact(eval_dataset_artifact[0], eval_dataset_artifact[1])
        mlflow.log_artifact(golden_dataset_artifact[0], golden_dataset_artifact[1])
        logger.info("logged evaluation and golden datasets")
        
        # log the code files
        code_files = return_code_files()
        
        for code_file in code_files:
            mlflow.log_artifact(code_file, "code")
        logger.info("code files logged")
    
        # set tag for the run
        mlflow.set_tag("phase", "historical_threshold")
        
    # extract info from run
    run_id = run.info.run_id
    run_name = run.info.run_name
    
    # log to json file
    log_run_info(run_id, run_name)