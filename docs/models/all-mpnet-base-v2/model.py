import torch
from djl_python import Input, Output
from sentence_transformers import SentenceTransformer

_model = None


def _load():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"use_safetensors": True, "torch_dtype": torch.float16},
        )
        _model.to("cuda")
        _model.eval()
    return _model


def handle(inputs: Input) -> Output:
    if inputs.is_empty():
        return None
    model = _load()
    data = inputs.get_as_json()
    texts = data.get("inputs", []) if isinstance(data, dict) else data
    if isinstance(texts, str):
        texts = [texts]
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=max(1, len(texts)),
        )
    out = Output()
    out.add_as_json(embeddings.tolist())
    return out
