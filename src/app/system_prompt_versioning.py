from langfuse import get_client
from dotenv import load_dotenv

from src.configs.config import params_config

# load the api keys
load_dotenv()

chunk_size = params_config.rag_app.chunk_size
chunk_overlap = params_config.rag_app.chunk_overlap
output_dimensions = params_config.rag_app.embedding_dimensions
k = params_config.rag_app.k

system_prompt = """You are a helpful assistant. Answer the user query
based on the given context only. If you do not know the answer
say I don't know. Do not add any preamble to the response.
Always try to answer in simple language. make sure your answer sticks to the input and the available context,
and be factually correct"""

# add system prompt to langfuse

langfuse = get_client()

created_prompt = langfuse.create_prompt(
    name="rag_app_system_prompt",
    type="text",
    prompt=system_prompt,
    labels=["staging"],
    config={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_dims": output_dimensions,
        "k": k
    }
)

print(created_prompt.prompt)
print(created_prompt.version)
print(created_prompt.labels)