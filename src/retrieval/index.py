"""
Nearest-neighbour search over frame embeddings.

Two backends behind one call. Both are exact; the choice is about where the
memory lives, not about accuracy.

Sizing note, because it decides whether FAISS is worth the dependency: exact
search is one GEMM of (n_query x dim) against (dim x n_ref). A 2-hour film at
30fps is 216k frames, which at 512 dims is 442 MB in fp32 and about 0.2 GFLOP
per query -- a T4 does that in single-digit milliseconds. Approximate indexes
(IVF, HNSW, PQ) only start paying for themselves in the millions of frames, and
they trade away exactly the fine-grained precision this task depends on. Use
flat/exact until measurement says otherwise.
"""
from __future__ import annotations

import numpy as np
import torch


class TorchFlatIP:
    """Exact inner-product search on GPU. Rows are assumed L2-normalised."""

    def __init__(self, vectors: np.ndarray, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # fp16 halves bandwidth and costs nothing measurable in rank order at
        # these dimensions; the scores are still returned as fp32.
        self.ref = torch.from_numpy(vectors).to(self.device).half()
        self.n, self.dim = vectors.shape

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = torch.from_numpy(queries).to(self.device).half()
        scores = q @ self.ref.T
        k = min(k, self.n)
        vals, idx = torch.topk(scores.float(), k, dim=1)
        return vals.cpu().numpy(), idx.cpu().numpy()

    def full_scores(self, queries: np.ndarray) -> np.ndarray:
        q = torch.from_numpy(queries).to(self.device).half()
        return (q @ self.ref.T).float().cpu().numpy()


class FaissFlatIP:
    """Exact inner-product search via FAISS, for parity checks and CPU use."""

    def __init__(self, vectors: np.ndarray, use_gpu: bool = False):
        import faiss

        self.n, self.dim = vectors.shape
        index = faiss.IndexFlatIP(self.dim)
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(
            np.ascontiguousarray(queries, dtype=np.float32), min(k, self.n))

    def full_scores(self, queries: np.ndarray) -> np.ndarray:
        return self.search(queries, self.n)[0]


def build_index(vectors: np.ndarray, backend: str = "torch", device: str = "cuda"):
    if backend == "torch":
        return TorchFlatIP(vectors, device)
    if backend == "faiss":
        return FaissFlatIP(vectors)
    raise KeyError(f"unknown backend {backend!r}")
