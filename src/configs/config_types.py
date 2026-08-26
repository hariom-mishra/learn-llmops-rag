from pydantic import BaseModel, ConfigDict


class RAGAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_label: str
    llm: str
    embedding_model: str
    embedding_dimensions: int
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    k: int
    tokenizer_encoding: str
    search_type: str


class FiltrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filtration_threshold: float
    max_retries: int
    filtration_critic_model: str


class EvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_evolution: int
    evolutions: dict[str, float]


class ContextConstructionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_critic_model: str
    max_contexts_per_document: int
    context_threshold: float
    max_context_retries: int
    context_chunk_size: int
    context_chunk_overlap: int


class GoldenDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_model: str
    max_golden_per_context: int
    golden_dataset_filename: str
    filtration_config: FiltrationConfig
    evolution_config: EvolutionConfig
    constext_construction_config: ContextConstructionConfig


class AsyncConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    throttle_value: int
    max_concurrent: int


class DisplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_dir: str
    report_dir: str


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge_llm: str
    async_config: AsyncConfig
    display_config: DisplayConfig


class EvaluationDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_dataset_filename: str


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rag_app: RAGAppConfig
    golden_dataset: GoldenDatasetConfig
    evaluation: EvaluationConfig
    evaluation_dataset: EvaluationDatasetConfig

