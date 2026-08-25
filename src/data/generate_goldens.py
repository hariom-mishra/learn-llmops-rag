from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig, EvolutionConfig, ContextConstructionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DOCS_PATH = ROOT_PATH / "data" / "processed"

# FIXED: Changed argument type to Path or str, and fixed the logic
def get_dir_path(provided_path: Path | str) -> list[str]:
    dir_path = Path(provided_path)

    # FIXED: Added () to is_dir()
    if dir_path.exists() and dir_path.is_dir():
        paths = dir_path.glob("*.txt")
        return [path.as_posix() for path in paths] 
    
    # FIXED: Return an empty list instead of None if path doesn't exist
    return []

# filteration config 
filtration_config = FiltrationConfig(
    synthetic_input_quality_threshold=0.6,
    max_quality_retries=2,
    critic_model='gpt-5-mini' # Note: Assuming 'gpt-5-min' was a typo for 'mini'
)

# evolution
evolution_config = EvolutionConfig(
    evolutions={
        Evolution.MULTICONTEXT: 0.1,
        Evolution.CONCRETIZING: 0.3,
        Evolution.CONSTRAINED: 0.4,
        Evolution.COMPARATIVE: 0.2
    },
    num_evolutions=2
)

# context - used for generation from document to goldens
context_config = ContextConstructionConfig(
    critic_model="gpt-5.4-mini",
    max_contexts_per_document=2,
    context_quality_threshold=0.7, 
)

# create the synthesizer
synthesizer = Synthesizer(
    model='gpt-5.4-mini',
    filtration_config=filtration_config,
    evolution_config=evolution_config
)

# generate goldens
# Make sure your DOCS_PATH actually has .txt files, otherwise this will return empty
document_list = get_dir_path(DOCS_PATH)

if not document_list:
    print(f"Warning: No .txt files found in {DOCS_PATH.resolve()}")
else:
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_list,
        max_goldens_per_context=3,
        include_expected_output=True,
        context_construction_config=context_config
    )

    # save the goldens
    GOLDEN_PATH = ROOT_PATH / "data" / "evaluation" / "goldens"
    GOLDEN_PATH.mkdir(exist_ok=True, parents=True)

    synthesizer.save_as(
        file_name="golden_dataset",
        file_type="json",
        directory=GOLDEN_PATH.as_posix()
    )
    print("Goldens saved successfully!")