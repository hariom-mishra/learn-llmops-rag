from rag_workflow import graph
from pathlib import Path
from dotenv import load_dotenv
from time import sleep

from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.dataset.dataset import EvaluationDataset

load_dotenv()

ROOT_DIR = Path()

GOLDEN_PATH = ROOT_DIR / "datasets" / "goldens" / "golden_dataset.json"

EVAL_DATA_DIRECTORY = GOLDEN_PATH.parent.parent / "eval_dataset"

EVAL_DATA_DIRECTORY.mkdir(exist_ok=True, parents=True)

dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path=GOLDEN_PATH.as_posix()
    )

for golden in dataset.goldens:
    final_state = graph.invoke({"query": golden.input})
    sleep(3)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=final_state.get("response"),
        expected_output=golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state["retrieved_documents"]]
    )
    dataset.add_test_case(
        test_case=test_case
    )


dataset.save_as(
    file_name="eval_dataset",
    file_type="json",
    directory=EVAL_DATA_DIRECTORY,
    include_test_cases=True
)