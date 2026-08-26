from src.configs.config import params_config
from src.data.generate_eval_dataset import generate_evaluation_dataset
from src.evals.application_evals.evaluate_rag_app import evaluate_app
from dotenv import load_dotenv
from pathlib import Path
from langfuse import get_client

load_dotenv()

#to compare the params we have to remove nesting 
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

#get the latest evaluation results
def get_latest_result(path: Path):
    filenames: []

    files = path.glob("*.json")

    for file in files:
        filenames.append(file.stem)
    recent_file = max(filenames)

    return recent_file

#extract the metrics from the result json file
def get_metrics_from_results(result_json) -> dict:
    metrics = {}
    with open(result_json, "r") as file:
        metrics_result = json.load(file)

    for result in metrics_result:
        metric_name = result["metric"].removesuffix("[GEval]") if "[GEval]" in result["metric"] else result["metric"]
        scores = result["scores"]

        avg_score = round((sum(scores) / len(scores)), 2)

        metrics[metric_name] = avg_score

    return metrics


def import_system_prompt(label="staging") -> str:
    langfuse = get_client()
    prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=label
    )

    return prompt
