from huggingface_hub import InferenceClient

from app.core.config import HF_TOKEN
from app.core.supabase import get_supabase_admin

CHUNKS_TABLE = "resume_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy singletons — neither loads a model into this process at import time.
# Hosted inference runs on Hugging Face infra; the local fallback
# (FastEmbed/ONNX, same model, same 384-dim output) loads a ~90MB model
# on first use only.
_hf_client: InferenceClient | None = None
_local_model = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(token=HF_TOKEN or None)
    return _hf_client


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding

        _local_model = TextEmbedding(EMBEDDING_MODEL)
    return [[float(x) for x in vector] for vector in _local_model.embed(texts)]


def _embed(texts: list[str]) -> list[list[float]]:
    try:
        vectors = _get_hf_client().feature_extraction(texts, model=EMBEDDING_MODEL)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
    except Exception as exc:
        print(f"[embedder_service] hosted inference failed ({exc}); trying local fallback")
    try:
        return _embed_local(texts)
    except ImportError:
        raise RuntimeError(
            "Resume embeddings are unavailable: hosted Hugging Face inference failed "
            "and the local fallback is not installed. Run `pip install fastembed` "
            "or check your HF_TOKEN credits."
        )


def embed_and_store(file_id: str, filename: str, chunks: list[str]) -> int:
    """Embed chunks and persist them in Supabase (pgvector). Returns number of chunks stored."""
    if not chunks:
        return 0

    embeddings = _embed(chunks)

    rows = [
        {
            "file_id": file_id,
            "filename": filename,
            "chunk_index": i,
            "content": chunk,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    get_supabase_admin().table(CHUNKS_TABLE).insert(rows).execute()
    return len(chunks)


def delete_chunks(file_id: str) -> None:
    try:
        get_supabase_admin().table(CHUNKS_TABLE).delete().eq("file_id", file_id).execute()
    except Exception as exc:
        print(f"[embedder_service] delete_chunks error for {file_id}: {exc}")


def query_resume(file_id: str, query: str, n_results: int = 5) -> list[str]:
    query_embedding = _embed([query])[0]
    result = (
        get_supabase_admin()
        .rpc(
            "match_resume_chunks",
            {
                "query_embedding": query_embedding,
                "match_file_id": file_id,
                "match_count": n_results,
            },
        )
        .execute()
    )
    return [row["content"] for row in result.data]
