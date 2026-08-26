from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.evaluate import evaluate
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from pathlib import Path
from dotenv import load_dotenv

from src.configs.config import params_config

load_dotenv()

# Settings from config
JUDGE_LLM = params_config.evaluation.judge_llm
THROTTLE_VALUE = params_config.evaluation.async_config.throttle_value
MAX_CONCURRENT = params_config.evaluation.async_config.max_concurrent
RESULTS_DIR = params_config.evaluation.display_config.result_dir
REPORT_DIR = params_config.evaluation.display_config.report_dir
EVALUATION_DATASET_FILENAME = params_config.evaluation_dataset.evaluation_dataset_filename

model = JUDGE_LLM

# define the metrics
recall = ContextualRecallMetric(model=model)
precision = ContextualPrecisionMetric(model=model)
contextual_relevancy = ContextualRelevancyMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model)
faithfulness = FaithfulnessMetric(model=model)

# define the custom metrics
answer_correctness = GEval(
    name="answer correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare
    the 'expected_output' with the 'actual_output'. Penalize wrong facts""",
    rubric=[Rubric(score_range=(0,5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has minor differences"),
            Rubric(score_range=(10,10), expected_outcome=r"100% correct")],
    model=model
)

simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read the 'actual output' first and then the 'input'",
        "Check whether the response is simple and easy to understand or not",
        "Make sure the response has least number of technical jargons and is student friendly"
    ],
    rubric=[
        Rubric(score_range=(0,3), expected_outcome="Too Difficult"),
        Rubric(score_range=(4,7), expected_outcome="Slightly difficult"),
        Rubric(score_range=(8,9), expected_outcome="moderately simple"),
        Rubric(score_range=(10,10), expected_outcome="very simple")
    ],
    model=model
)

def evaluate_app():

    # define the dataset path
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_PATH = (ROOT_DIR / "data" / "evaluation" / "eval_dataset" / EVALUATION_DATASET_FILENAME).with_suffix(".json")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {DATASET_PATH}")

    # ensure result and report directories exist
    results_folder = ROOT_DIR / "reports" / RESULTS_DIR
    file_output_dir = ROOT_DIR / "reports" / REPORT_DIR
    results_folder.mkdir(parents=True, exist_ok=True)
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # load the dataset
    dataset = EvaluationDataset()
    
    # load the test cases
    dataset.add_test_cases_from_json_file(
        file_path=DATASET_PATH,
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        retrieval_context_key_name="retrieval_context"
    )
    
    # store the test cases in a list
    test_cases = dataset.test_cases
    
    # evaluate the dataset
    return evaluate(
        test_cases=test_cases,
        metrics=[
            recall,
            precision,
            answer_relevancy,
            faithfulness,
            contextual_relevancy,
            answer_correctness,
            simple_explanation
        ],
        async_config=AsyncConfig(
            throttle_value=THROTTLE_VALUE,
            max_concurrent=MAX_CONCURRENT
        ),
        display_config=DisplayConfig(
            results_folder=results_folder.as_posix(),
            file_type="md",
            file_output_dir=file_output_dir.as_posix()
        )
    )


if __name__ == "__main__":
    evaluate_app()