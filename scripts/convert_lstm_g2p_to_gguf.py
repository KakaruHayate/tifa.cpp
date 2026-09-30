#!/usr/bin/env python3
"""Convert the TIFA English LSTM G2P ONNX pair into a single GGUF file.

Usage:
    python convert_lstm_g2p_to_gguf.py \\
        --model-dir ../../models/TIFA-1.0-ST/assets/LstmG2p-Eng \\
        -o models/lstm-g2p-eng.gguf

Inputs (all required, in `--model-dir`):
    encoder.onnx    2-layer bidirectional LSTM (char embedding + projection)
    decoder.onnx    2-layer LSTM with attention over the encoder outputs
    char.json       grapheme vocabulary
    phonemes.json   phoneme vocabulary

Output:
    one GGUF file containing
      * architecture metadata          (general.architecture = "lstm-g2p")
      * hyper-parameters               (lstm_g2p.*: hidden sizes, layer counts,
                                        max decode length, beam size, special ids)
      * the two vocabularies           (lstm_g2p.char_vocab / .phoneme_vocab, as
                                        the raw JSON text)
      * every weight, dequantised to F32 and re-laid-out for the C++ consumer,
        keyed by its ONNX node path (`encoder.lstm.0.weight`, ...)

The exported graphs store their LSTM weights with the ONNX Runtime
"transposed" layout ([num_directions, input_size, 4*hidden] and
[num_directions, hidden_size, 4*hidden]) and quantise them with
`DynamicQuantizeLSTM` / `MatMulInteger`.  Every quantised initializer is
dequantised here as `(q - zero_point) * scale` (the zero points are all 0 for
the LSTM/MatMul weights, the two embeddings use uint8 with a non-zero point),
and the weights are written out in the *output-major* layout the host GEMM
wants: `y[o] = dot(w[o], x)`.

The converter is self-contained: it needs only numpy, onnx and gguf.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np

try:
    import gguf  # type: ignore
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "error: the 'gguf' Python package is required "
        "(pip install -r scripts/requirements.txt).\n"
    )
    raise

try:
    import onnx  # type: ignore
    from onnx import numpy_helper  # type: ignore
except ImportError:  # pragma: no cover
    sys.stderr.write("error: the 'onnx' Python package is required (pip install onnx).\n")
    raise

log = logging.getLogger("convert_lstm_g2p_to_gguf")

ARCH = "lstm-g2p"
ARCH_VERSION = 1

# ---------------------------------------------------------------------------
# graph -> gguf tensor name table
# ---------------------------------------------------------------------------

# Every parameter-bearing initializer of encoder.onnx / decoder.onnx, mapped to
# the GGUF name the C++ side binds.  `transform` describes what has to happen
# on the way out:
#   "as-is"          keep the ONNX shape
#   "out-major"      transpose a [in, out] linear weight to [out, in]
#   "lstm-weight"    [D, in, 4H]        -> [D, 4H, in]  (drop D when it is 1)
#   "lstm-recur"     [D, hidden, 4H]    -> [D, 4H, hidden]
_TENSORS: list[tuple[str, str, str]] = [
    # ---- encoder -----------------------------------------------------------
    ("encoder", "encoder.embedding.weight_quantized", "encoder.embed.weight", "as-is"),
    ("encoder", "onnx::LSTM_391_quantized", "encoder.lstm.0.weight", "lstm-weight"),
    ("encoder", "onnx::LSTM_392_quantized", "encoder.lstm.0.recurrence", "lstm-recur"),
    ("encoder", "onnx::LSTM_390", "encoder.lstm.0.bias", "as-is"),
    ("encoder", "onnx::LSTM_434_quantized", "encoder.lstm.1.weight", "lstm-weight"),
    ("encoder", "onnx::LSTM_435_quantized", "encoder.lstm.1.recurrence", "lstm-recur"),
    ("encoder", "onnx::LSTM_433", "encoder.lstm.1.bias", "as-is"),
    ("encoder", "onnx::MatMul_438_quantized", "encoder.projection.weight", "out-major"),
    ("encoder", "encoder.projection.bias", "encoder.projection.bias", "as-is"),
    # ---- decoder -----------------------------------------------------------
    ("decoder", "decoder.embedding.weight_quantized", "decoder.embed.weight", "as-is"),
    ("decoder", "onnx::LSTM_249_quantized", "decoder.lstm.0.weight", "lstm-weight"),
    ("decoder", "onnx::LSTM_250_quantized", "decoder.lstm.0.recurrence", "lstm-recur"),
    ("decoder", "onnx::LSTM_251", "decoder.lstm.0.bias", "as-is"),
    ("decoder", "onnx::LSTM_269_quantized", "decoder.lstm.1.weight", "lstm-weight"),
    ("decoder", "onnx::LSTM_270_quantized", "decoder.lstm.1.recurrence", "lstm-recur"),
    ("decoder", "onnx::LSTM_271", "decoder.lstm.1.bias", "as-is"),
    ("decoder", "onnx::MatMul_230_quantized", "decoder.attention.key.weight", "out-major"),
    ("decoder", "decoder.attention.attention.bias", "decoder.attention.key.bias", "as-is"),
    ("decoder", "onnx::MatMul_231_quantized", "decoder.attention.value.weight", "out-major"),
    ("decoder", "onnx::MatMul_272_quantized", "decoder.fc.0.weight", "out-major"),
    ("decoder", "decoder.fc.0.bias", "decoder.fc.0.bias", "as-is"),
    ("decoder", "onnx::MatMul_273_quantized", "decoder.fc.3.weight", "out-major"),
    ("decoder", "decoder.fc.3.bias", "decoder.fc.3.bias", "as-is"),
]

# `_TENSORS` only lists the *quantised* multiplications.  These are the
# activations/attributes the C++ hard-codes; the shapes below are checked
# against the file so a re-export with different sizes fails loudly.
# All DynamicQuantizeLSTM nodes in both files are iofc, forward/bidirectional,
# with no peepholes, no clipping and no input_forget.
_EXPECTED_LSTM_ATTRS = {"hidden_size", "direction"}


def _node_key(name: str) -> str:
    return name.lstrip("/")


def load_quantizers(path: pathlib.Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return the initializers of *path* keyed by name, plus the LSTM node paths."""
    model = onnx.load(str(path))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    lstm_nodes: list[str] = []
    for node in model.graph.node:
        if node.op_type != "DynamicQuantizeLSTM":
            continue
        lstm_nodes.append(_node_key(node.name))
        for attr in node.attribute:
            name = attr.name
            if name == "activations":
                values = [s.decode() for s in attr.strings]
                expected = ["Sigmoid", "Tanh", "Tanh"]
                if values != expected and values != expected + expected:
                    raise SystemExit(f"error: {node.name}: unsupported activations {values}")
                continue
            if name == "direction":
                value = attr.s.decode()
                if value not in ("forward", "bidirectional"):
                    raise SystemExit(f"error: {node.name}: unsupported direction {value}")
                continue
            if name == "hidden_size":
                continue
            raise SystemExit(f"error: {node.name}: unexpected LSTM attribute '{name}'")
    return init, lstm_nodes


def dequantize(init: dict[str, np.ndarray], file_key: str, name: str) -> np.ndarray:
    """`(q - zero_point) * scale`, with per-direction scales when present."""
    quant = init[name]
    if not name.endswith("_quantized"):
        # a plain float initializer (biases, …): nothing to dequantise
        return quant.astype(np.float32)
    scale_name = name.replace("_quantized", "_scale")
    zp_name = name.replace("_quantized", "_zero_point")
    if scale_name not in init:
        raise SystemExit(f"error: {name}: quantized without a scale")
    scale = init[scale_name].astype(np.float32)
    zp = init[zp_name]
    q = quant.astype(np.float32)
    if q.ndim == 3:
        if scale.size != q.shape[0] or zp.size != q.shape[0]:
            raise SystemExit(f"error: {name}: scale/zero_point do not match the leading axis")
        return (q - zp.astype(np.float32)[:, None, None]) * scale[:, None, None]
    if scale.size != 1:
        raise SystemExit(f"error: {name}: unsupported per-channel scale {scale.shape}")
    return (q - np.float32(zp.reshape(-1)[0])) * np.float32(scale.reshape(-1)[0])


def convert_weight(arr: np.ndarray, transform: str, name: str) -> np.ndarray:
    if transform == "as-is":
        return np.ascontiguousarray(arr, dtype=np.float32)
    if transform == "out-major":
        if arr.ndim != 2:
            raise SystemExit(f"error: {name}: expected a 2-D weight, got {arr.shape}")
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    if transform in ("lstm-weight", "lstm-recur"):
        if arr.ndim != 3:
            raise SystemExit(f"error: {name}: expected [directions, in, 4*hidden], got {arr.shape}")
        out = np.ascontiguousarray(np.transpose(arr, (0, 2, 1)), dtype=np.float32)
        return out[0] if out.shape[0] == 1 else out
    raise SystemExit(f"error: {name}: unknown transform '{transform}'")


def graph_params(enc_init: dict[str, np.ndarray], dec_init: dict[str, np.ndarray],
                 encoder_lstm: list[str], decoder_lstm: list[str]) -> dict[str, Any]:
    """Read the hyper-parameters off the exported tensors (never guessed)."""
    if len(encoder_lstm) != 2 or len(decoder_lstm) != 2:
        raise SystemExit("error: expected two LSTM layers in each graph")
    # The two graphs share the ONNX-export initializer namespace, so every
    # lookup below states which file it comes from.
    init: dict[str, np.ndarray] = {**enc_init, **dec_init}

    emb_c = init["encoder.embedding.weight_quantized"]
    emb_p = init["decoder.embedding.weight_quantized"]
    w0 = init["onnx::LSTM_391_quantized"]      # [2, input, 4*hidden]
    r0 = init["onnx::LSTM_392_quantized"]      # [2, hidden, 4*hidden]
    w1 = init["onnx::LSTM_434_quantized"]
    proj = init["onnx::MatMul_438_quantized"]  # [2*hidden, project]

    hidden = r0.shape[1]
    input_dim = w0.shape[1]
    if r0.shape[2] != 4 * hidden or w0.shape[2] != 4 * hidden:
        raise SystemExit("error: LSTM weights are not in the transposed ONNX layout")
    if w1.shape[1] != 2 * hidden:
        raise SystemExit("error: encoder layer 1 does not consume both directions")
    if proj.shape[0] != 2 * hidden:
        raise SystemExit("error: encoder projection does not consume both directions")
    if emb_c.shape[1] != input_dim:
        raise SystemExit("error: char embedding width does not match the LSTM input")

    dw0 = init["onnx::LSTM_249_quantized"]     # [1, emb+hidden, 4*hidden]
    dr0 = init["onnx::LSTM_250_quantized"]
    dw1 = init["onnx::LSTM_269_quantized"]
    key = init["onnx::MatMul_230_quantized"]   # [2*hidden, hidden]
    val = init["onnx::MatMul_231_quantized"]   # [hidden, 1]
    if dw0.shape[1] != emb_p.shape[1] + hidden:
        raise SystemExit("error: decoder LSTM input is not embedding+context")
    if dr0.shape[1] != hidden or dw1.shape[1] != hidden:
        raise SystemExit("error: decoder LSTM hidden size mismatch")
    if key.shape[0] != 2 * hidden or val.shape[0] != hidden or val.shape[1] != 1:
        raise SystemExit("error: unexpected attention shapes")

    return {
        "hidden": int(hidden),
        "input_dim": int(input_dim),
        "encoder_layers": 2,
        "encoder_bidirectional": True,
        "decoder_layers": 2,
        "embedding_dim": int(emb_p.shape[1]),
        "num_phonemes": int(emb_p.shape[0]),
        "num_chars": int(emb_c.shape[0]),
        "max_len": 48,                 # g2p/converters/lstm.py:_max_len
        "beam_size": 16,               # converter default
    }


def special_ids(vocab: dict[str, int], names: tuple[str, ...]) -> dict[str, int]:
    out = {}
    for name in names:
        if name not in vocab:
            raise SystemExit(f"error: vocabulary is missing '{name}'")
        out[name] = int(vocab[name])
    return out


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def convert(model_dir: pathlib.Path, output: pathlib.Path, *, dict_path: str | None,
            beam_size: int, max_len: int) -> None:
    enc_path  = model_dir / "encoder.onnx"
    dec_path  = model_dir / "decoder.onnx"
    char_path = model_dir / "char.json"
    phon_path = model_dir / "phonemes.json"
    for p in (enc_path, dec_path, char_path, phon_path):
        if not p.exists():
            raise SystemExit(f"error: {p} not found")

    enc_init, enc_lstm = load_quantizers(enc_path)
    dec_init, dec_lstm = load_quantizers(dec_path)
    char_vocab = json.loads(char_path.read_text(encoding="utf-8"))
    phon_vocab = json.loads(phon_path.read_text(encoding="utf-8"))

    gp = graph_params(enc_init, dec_init, enc_lstm, dec_lstm)
    gp["beam_size"] = int(beam_size)
    gp["max_len"] = int(max_len)
    if max(phon_vocab.values()) >= gp["num_phonemes"]:
        raise SystemExit("error: phoneme vocabulary is larger than the decoder embedding")

    # `encoder.embedding.weight_quantized` carries a <unk> row the LSTM indexer
    # relies on; the C++ reads ids straight out of the JSON vocabularies.
    for vocab, key, size in ((char_vocab, "char", gp["num_chars"]),
                             (phon_vocab, "phoneme", gp["num_phonemes"])):
        if max(vocab.values()) >= size:
            raise SystemExit(f"error: {key} vocabulary is larger than the embedding table")

    writer = gguf.GGUFWriter(output, ARCH)
    writer.add_name(output.stem)
    writer.add_string("general.version", str(ARCH_VERSION))
    writer.add_string("lstm_g2p.source", enc_path.name + "+" + dec_path.name)
    for key in ("hidden", "input_dim", "encoder_layers", "decoder_layers", "embedding_dim",
                "num_chars", "num_phonemes", "max_len", "beam_size"):
        writer.add_int32(f"lstm_g2p.{key}", int(gp[key]))
    writer.add_bool("lstm_g2p.encoder_bidirectional", bool(gp["encoder_bidirectional"]))
    ids = special_ids(phon_vocab, ("<unk>", "<pad>", "<bos>", "<eos>"))
    for name, value in ids.items():
        writer.add_int32(f"lstm_g2p.phoneme_{name.strip('<>')}", value)
    writer.add_int32("lstm_g2p.char_unk", int(char_vocab.get("<unk>", 0)))
    if dict_path:
        writer.add_string("lstm_g2p.dict_path", dict_path)
    # vocabularies ride along as raw JSON so the C++ parses them with the same
    # reader the rest of the config uses.
    writer.add_string("lstm_g2p.char_vocab", json.dumps(char_vocab, ensure_ascii=False))
    writer.add_string("lstm_g2p.phoneme_vocab", json.dumps(phon_vocab, ensure_ascii=False))

    written: dict[str, tuple[int, ...]] = {}
    total = 0
    for file_key, src, dst, transform in _TENSORS:
        init = enc_init if file_key == "encoder" else dec_init
        if src not in init:
            raise SystemExit(f"error: {file_key}.onnx has no initializer '{src}'")
        arr = convert_weight(dequantize(init, file_key, src), transform, dst)
        if dst in written:
            raise SystemExit(f"error: duplicate tensor name '{dst}'")
        written[dst] = arr.shape
        writer.add_tensor(dst, np.ascontiguousarray(arr, dtype=np.float32),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
        total += arr.nbytes

    # every quantised initializer must have been consumed: the _TENSORS table is
    # the only place a layout decision is made, so a missed one is a silent bug.
    leftovers = []
    for file_key, init in (("encoder", enc_init), ("decoder", dec_init)):
        for name in init:
            if name.endswith("_quantized") and not any(s == name for f, s, _, _ in _TENSORS if f == file_key):
                leftovers.append(f"{file_key}.{name}")
    if leftovers:
        raise SystemExit("error: quantised initializers were not converted: " + ", ".join(leftovers))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    log.info("encoder: %d layers, hidden %d, input %d, bidirectional",
             gp["encoder_layers"], gp["hidden"], gp["input_dim"])
    log.info("decoder: %d layers, embedding %d, %d phonemes",
             gp["decoder_layers"], gp["embedding_dim"], gp["num_phonemes"])
    log.info("chars %d, phonemes %d, max_len %d, beam %d",
             gp["num_chars"], gp["num_phonemes"], gp["max_len"], gp["beam_size"])
    log.info("wrote %s (%.1f MB, %d tensors)",
             output, output.stat().st_size / 1e6, len(written))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=pathlib.Path, required=True,
                        help="directory holding encoder.onnx / decoder.onnx / the vocabs")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="output .gguf path")
    parser.add_argument("--dict-path", default=None,
                        help="optional pronunciation dictionary path recorded in the gguf")
    parser.add_argument("--beam-size", type=int, default=16,
                        help="beam width recorded in the gguf (default: 16)")
    parser.add_argument("--max-len", type=int, default=48,
                        help="decode step cap recorded in the gguf (default: 48)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    convert(args.model_dir, args.output, dict_path=args.dict_path,
            beam_size=args.beam_size, max_len=args.max_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
