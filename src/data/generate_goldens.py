from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig, EvolutionConfig, ContextConstructionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path

from src.configs.config import params_config

load_dotenv()


def generate_golden_dataset():
    ROOT_PATH = Path(__file__).resolve().parent.parent.parent
    DOCS_PATH = ROOT_PATH / "data" / "processed"

    def get_dir_path(provided_path: Path | str) -> list[str]:
        dir_path = Path(provided_path)

        if dir_path.exists() and dir_path.is_dir():
            paths = dir_path.glob("*.txt")
            return [path.as_posix() for path in paths] 
        
        return []

    # filteration config 
    filtration_config = FiltrationConfig(
        synthetic_input_quality_threshold=params_config.golden_dataset.filtration_config.filtration_threshold,
        max_quality_retries=params_config.golden_dataset.filtration_config.max_retries,
        critic_model=params_config.golden_dataset.filtration_config.filtration_critic_model
    )

    # evolution
    evolutions_dict = {
        Evolution[k.upper()]: v
        for k, v in params_config.golden_dataset.evolution_config.evolutions.items()
    }
    evolution_config = EvolutionConfig(
        evolutions=evolutions_dict,
        num_evolutions=params_config.golden_dataset.evolution_config.num_evolution
    )

    # context - used for generation from document to goldens
    context_cfg = params_config.golden_dataset.constext_construction_config
    context_config = ContextConstructionConfig(
        critic_model=context_cfg.context_critic_model,
        max_contexts_per_document=context_cfg.max_contexts_per_document,
        context_quality_threshold=context_cfg.context_threshold,
        max_retries=context_cfg.max_context_retries,
        chunk_size=context_cfg.context_chunk_size,
        chunk_overlap=context_cfg.context_chunk_overlap
    )

    # create the synthesizer
    synthesizer = Synthesizer(
        model=params_config.golden_dataset.dataset_model,
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
            max_goldens_per_context=params_config.golden_dataset.max_golden_per_context,
            include_expected_output=True,
            context_construction_config=context_config
        )

        # save the goldens
        GOLDEN_PATH = ROOT_PATH / "data" / "evaluation" / "goldens"
        GOLDEN_PATH.mkdir(exist_ok=True, parents=True)

        synthesizer.save_as(
            file_name=params_config.golden_dataset.golden_dataset_filename,
            file_type="json",
            directory=GOLDEN_PATH.as_posix()
        )
        print("Goldens saved successfully!")


if __name__ == "__main__":
    generate_golden_dataset()