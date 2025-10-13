from sentence_transformers import SentenceTransformer
import torch
from typing import Union, List

@torch.inference_mode()
def get_lang_emb(text: Union[str, List[str]]) -> torch.Tensor:
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and lightweight
    if isinstance(text, str):
        text = [text]
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings if len(embeddings) > 1 else embeddings[0]
