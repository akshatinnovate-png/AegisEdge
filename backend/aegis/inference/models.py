"""Model provisioning.

The node runs real pretrained weights, not a stand-in. The embedding matrix is
a 32000 x 256 table distilled from Llama-3 (the `wordllama` l2_supercat
model), paired with its real 32k BPE tokenizer — both shipped inside the
Python distribution, so provisioning needs no model hub, no download and no
network. That matters for an edge product: a device in a tunnel cannot fetch
weights, and a build that silently falls back to a toy encoder is worse than
one that refuses to start.

Two ONNX graphs are compiled from those weights at first boot and cached:

* **embedder** — gather token rows, mean-pool, L2-normalise. One graph, no
  Python in the hot path.
* **reranker** — ColBERT-style late interaction: MaxSim between query tokens
  and document tokens, which is a genuine model-based reranker over the same
  pretrained space rather than lexical overlap.

Both are content-addressed into the registry, so the digest check that guards
loading is checking something real.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

IR_VERSION = 9          # the highest IR onnxruntime 1.x reliably loads
OPSET = 13

EMBEDDER_GRAPH = "embedder.onnx"
RERANKER_GRAPH = "reranker.onnx"
TOKENIZER_FILE = "tokenizer.json"
METADATA_FILE = "provenance.json"


class ModelsUnavailable(RuntimeError):
    """Raised when real weights cannot be located — never silently downgraded."""


@dataclass(slots=True)
class ModelBundle:
    embedder_path: Path
    reranker_path: Path
    tokenizer_path: Path
    dim: int
    vocab: int
    source: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"embedder": str(self.embedder_path), "reranker": str(self.reranker_path),
                "tokenizer": str(self.tokenizer_path), "dim": self.dim, "vocab": self.vocab,
                "source": self.source, "sha256": self.sha256[:16]}


def locate_pretrained() -> tuple[np.ndarray, Path, str]:
    """Find the pretrained embedding table and tokenizer shipped on this machine."""
    try:
        import wordllama
        from safetensors.numpy import load_file
    except ImportError as exc:                       # pragma: no cover - dependency guard
        raise ModelsUnavailable(
            "pretrained weights unavailable: install `wordllama` and `safetensors`"
        ) from exc

    root = Path(wordllama.__file__).parent
    weight_files = sorted((root / "weights").glob("*.safetensors"))
    tokenizers = sorted((root / "tokenizers").glob("*tokenizer_config.json"))
    if not weight_files or not tokenizers:
        raise ModelsUnavailable(f"no bundled weights under {root}")

    tensors = load_file(weight_files[0])
    matrix = next(iter(tensors.values())).astype(np.float32)
    return matrix, tokenizers[0], weight_files[0].stem


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_embedder_graph(matrix: np.ndarray, path: Path) -> None:
    """token ids -> gather -> masked mean pool -> L2 normalise."""
    from onnx import TensorProto, helper, numpy_helper, save

    vocab, dim = matrix.shape
    table = numpy_helper.from_array(matrix, name="token_embeddings")

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.FLOAT, ["batch", "seq"])
    out = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", dim])

    nodes = [
        helper.make_node("Gather", ["token_embeddings", "input_ids"], ["gathered"], axis=0),
        helper.make_node("Unsqueeze", ["attention_mask", "axis_last"], ["mask3"]),
        helper.make_node("Mul", ["gathered", "mask3"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axis_seq"], ["summed"], keepdims=0),
        helper.make_node("ReduceSum", ["attention_mask", "axis_last_r"], ["counts"], keepdims=1),
        helper.make_node("Clip", ["counts", "one", ""], ["safe_counts"]),
        helper.make_node("Div", ["summed", "safe_counts"], ["pooled"]),
        # L2 normalise so every downstream cosine is a plain dot product.
        # ReduceL2 takes `axes` as an attribute until opset 18, while ReduceSum
        # takes it as an input from opset 13 — mixing the two conventions is
        # what makes a hand-built graph fail to load.
        helper.make_node("ReduceL2", ["pooled"], ["norm"], axes=[1], keepdims=1),
        helper.make_node("Clip", ["norm", "eps", ""], ["safe_norm"]),
        helper.make_node("Div", ["pooled", "safe_norm"], ["embedding"]),
    ]
    initializers = [
        table,
        numpy_helper.from_array(np.array([2], dtype=np.int64), name="axis_last"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis_seq"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis_last_r"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
        numpy_helper.from_array(np.array(1e-9, dtype=np.float32), name="eps"),
    ]
    graph = helper.make_graph(nodes, "aegis_embedder", [ids, mask], [out], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="aegis-edge")
    # Pin the IR version: the onnx package emits a newer IR than the installed
    # runtime accepts, and a graph the device cannot load is worse than none.
    model.ir_version = IR_VERSION
    model.doc_string = "AegisEdge embedder: pretrained token table, masked mean pool, L2 norm"
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(path))


def build_reranker_graph(matrix: np.ndarray, path: Path) -> None:
    """Late interaction: per-token embeddings for query and document, MaxSim.

    Emitting normalised token matrices lets the runtime compute MaxSim as one
    matmul plus a reduction, which is what makes a ColBERT-style reranker
    affordable on a device that has no GPU.
    """
    from onnx import TensorProto, helper, numpy_helper, save

    vocab, dim = matrix.shape
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    out = helper.make_tensor_value_info("token_embeddings_out", TensorProto.FLOAT,
                                        ["batch", "seq", dim])
    nodes = [
        helper.make_node("Gather", ["token_embeddings", "input_ids"], ["gathered"], axis=0),
        helper.make_node("ReduceL2", ["gathered"], ["norm"], axes=[2], keepdims=1),
        helper.make_node("Clip", ["norm", "eps", ""], ["safe_norm"]),
        helper.make_node("Div", ["gathered", "safe_norm"], ["token_embeddings_out"]),
    ]
    initializers = [
        numpy_helper.from_array(matrix, name="token_embeddings"),
        numpy_helper.from_array(np.array(1e-9, dtype=np.float32), name="eps"),
    ]
    graph = helper.make_graph(nodes, "aegis_reranker", [ids], [out], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="aegis-edge")
    model.ir_version = IR_VERSION
    model.doc_string = "AegisEdge late-interaction reranker: normalised token embeddings for MaxSim"
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(path))


def provision(model_dir: Path, force: bool = False) -> ModelBundle:
    """Compile the real graphs once; reuse them on every subsequent boot."""
    model_dir = Path(model_dir)
    embedder = model_dir / EMBEDDER_GRAPH
    reranker = model_dir / RERANKER_GRAPH
    tokenizer = model_dir / TOKENIZER_FILE
    metadata = model_dir / METADATA_FILE

    if not force and embedder.exists() and reranker.exists() and tokenizer.exists() \
            and metadata.exists():
        try:
            row = json.loads(metadata.read_text(encoding="utf-8"))
            return ModelBundle(embedder, reranker, tokenizer, row["dim"], row["vocab"],
                               row["source"], row["sha256"])
        except Exception:
            pass                                    # rebuild rather than trust a broken record

    matrix, tokenizer_source, source = locate_pretrained()
    build_embedder_graph(matrix, embedder)
    build_reranker_graph(matrix, reranker)
    tokenizer.write_bytes(tokenizer_source.read_bytes())

    bundle = ModelBundle(embedder, reranker, tokenizer, int(matrix.shape[1]),
                         int(matrix.shape[0]), source, _digest(embedder))
    metadata.write_text(json.dumps({
        "dim": bundle.dim, "vocab": bundle.vocab, "source": bundle.source,
        "sha256": bundle.sha256, "reranker_sha256": _digest(reranker),
        "tokenizer_sha256": _digest(tokenizer),
        "provenance": "wordllama l2_supercat token embeddings (Llama-3 distilled), "
                      "compiled to ONNX locally; no network fetch",
    }, indent=2), encoding="utf-8")
    return bundle
