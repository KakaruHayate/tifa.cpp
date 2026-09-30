#!/usr/bin/env python3
"""Golden predictions for the English LSTM G2P (reference implementation).

Ports g2p/converters/lstm.py:LSTMConverter._predict verbatim over ONNX Runtime
and dumps the beam-search results for a fixed word list to JSON, so the C++
port can be diffed against the reference.

Usage: python make_lstm_g2p_golden.py --model-dir <LstmG2p-Eng> -o golden.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class _Beam:
    __slots__ = ("tokens", "score", "hidden", "cell", "finished")

    def __init__(self, tokens, score, hidden, cell, finished=False):
        self.tokens = tokens
        self.score = score
        self.hidden = hidden
        self.cell = cell
        self.finished = finished

    @property
    def normalized_score(self):
        return self.score / max(1, len(self.tokens) - 1)


def predict(sess_enc, sess_dec, char_vocab, phoneme_vocab, word, beam_size=16, max_len=48):
    word = word.lower().strip()
    indices = [char_vocab.get(c, char_vocab.get("<unk>", 0)) for c in word]
    src = np.array([[phoneme_vocab["<bos>"]] + indices + [phoneme_vocab["<eos>"]]],
                   dtype=np.int64)

    encoder_outputs, hidden, cell = sess_enc.run(None, {"input_ids": src})

    beams = [_Beam([phoneme_vocab["<bos>"]], 0.0, hidden, cell)]
    unk = phoneme_vocab["<unk>"]
    pad = phoneme_vocab["<pad>"]
    bos = phoneme_vocab["<bos>"]
    eos = phoneme_vocab["<eos>"]
    idx_to_ph = {v: k for k, v in phoneme_vocab.items()}

    for _ in range(max_len):
        if all(b.finished for b in beams):
            break
        candidates = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            logits, hidden, cell, _ = sess_dec.run(None, {
                "decoder_input": np.array([[beam.tokens[-1]]], dtype=np.int64),
                "hidden": beam.hidden,
                "cell": beam.cell,
                "encoder_outputs": encoder_outputs,
            })
            log_probs = logits[0, 0, :].astype(np.float64)
            log_probs -= np.max(log_probs)
            log_probs -= np.log(np.exp(log_probs).sum())
            top_ids = np.argsort(-log_probs, kind="stable")[:beam_size]
            for pred_id in top_ids:
                pred_id = int(pred_id)
                candidates.append(_Beam(
                    tokens=beam.tokens + [pred_id],
                    score=beam.score + float(log_probs[pred_id]),
                    hidden=hidden, cell=cell,
                    finished=pred_id == eos))
        candidates.sort(key=lambda b: b.normalized_score, reverse=True)
        beams = candidates[:beam_size]

    def decode(pred_ids):
        return [idx_to_ph[i] for i in pred_ids if i not in (unk, pad, bos, eos)
                and i in idx_to_ph]

    out = []
    for beam in beams:
        pron = decode(beam.tokens[1:])
        if pron not in out:
            out.append(pron)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--beam-size", type=int, default=16)
    args = ap.parse_args()

    md = args.model_dir
    char_vocab = json.loads((md / "char.json").read_text(encoding="utf-8"))
    phoneme_vocab = json.loads((md / "phonemes.json").read_text(encoding="utf-8"))
    sess_enc = ort.InferenceSession(str(md / "encoder.onnx"))
    sess_dec = ort.InferenceSession(str(md / "decoder.onnx"))

    words = [
        "hello", "alignment", "phoneme", "tifa", "ggml", "xylem",
        "nhk", "helloworld", "separate", "different", "ka", "test",
        "yandex", "wohenhao", "openvpi", "diffsinger", "kakaru", " alignment",
    ]
    result = {}
    for w in words:
        result[w] = predict(sess_enc, sess_dec, char_vocab, phoneme_vocab, w,
                            beam_size=args.beam_size)
        print(w, "->", [" ".join(p) for p in result[w][:2]])

    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"wrote {args.output} ({len(result)} words)")


if __name__ == "__main__":
    main()
