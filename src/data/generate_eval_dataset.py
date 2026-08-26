from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter, INFO
from langfuse import get_client

from src.app.rag_workflow import graph
from src.configs.config import params_config

# dataset filenames
EVALUATION_DATASET_FILENAME = params_config.evaluation_dataset.evaluation_dataset_filename
GOLDEN_DATASET_FILENAME = params_config.golden_dataset.golden_dataset_filename

# load the api keys
load_dotenv()

def generate_evaluation_dataset():

    # create the logger
    logger = getLogger(name="Dataset Logger")
    # add stream handler
    handler = StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(INFO)
    # add formatter
    formatter = Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt=formatter)

    # create paths
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent

    GOLDENS_PATH = (ROOT_DIR / "data" / "evaluation" / "goldens" / GOLDEN_DATASET_FILENAME).with_suffix(".json")
    EVALUATION_DATA_DIR = ROOT_DIR / "data" / "evaluation" / "eval_dataset"

    # create dir
    EVALUATION_DATA_DIR.mkdir(exist_ok=True, parents=True)

    # dataset to read goldens from
    golden_dataset = EvaluationDataset()
    golden_dataset.add_goldens_from_json_file(file_path=GOLDENS_PATH)

    # dataset to hold produced test cases
    eval_dataset = EvaluationDataset()

    for count, golden in enumerate(golden_dataset.goldens, 1):
        final_state = graph.invoke({"query": golden.input})
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=final_state.get("response"),
            expected_output=golden.expected_output,
            retrieval_context=[doc.page_content for doc in final_state.get("retrieved_documents")]
        )
        eval_dataset.add_test_case(test_case=test_case)
        logger.log(level=INFO, msg=f"Added test case no. {count}")

    eval_dataset.save_as(
        file_type="json",
        directory=EVALUATION_DATA_DIR,
        file_name=EVALUATION_DATASET_FILENAME,
        include_test_cases=True
    )


if __name__ == "__main__":
    generate_evaluation_dataset()