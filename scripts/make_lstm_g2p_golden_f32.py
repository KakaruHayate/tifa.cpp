#!/usr/bin/env python3
"""F32 numpy reference for the English LSTM G2P, reading the converted GGUF.

The reference implementation (g2p/converters/lstm.py over ONNX Runtime) runs
its matmuls through int8 dynamic quantization; this port runs the same math in
plain float32 straight off the dequantized GGUF, which is what the C++ port
computes.  Goldens generated from this file are therefore the strict target
for the C++ differential test; the ONNX golden documents the quantized
reference behaviour (top-1 agreement only).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import gguf


def load(gguf_path: Path):
    m = gguf.GGUFReader(str(gguf_path))
    W = {}
    for t in m.tensors:
        W[t.name] = np.frombuffer(t.data, dtype=np.float32).astype(np.float32).reshape(
            tuple(t.shape)[::-1])

    def s(key):
        f = m.fields[key]
        return bytes(f.parts[f.data[0]]).decode()

    char_vocab = json.loads(s("lstm_g2p.char_vocab"))
    ph_vocab = json.loads(s("lstm_g2p.phoneme_vocab"))
    hidden = int(np.array(m.fields["lstm_g2p.hidden"].parts[-1]).reshape(-1)[0])
    return W, char_vocab, ph_vocab, hidden


class NumpyLstmG2p:
    def __init__(self, gguf_path: Path):
        self.W, self.char_vocab, self.ph_vocab, self.H = load(gguf_path)
        self.bos = self.ph_vocab["<bos>"]
        self.eos = self.ph_vocab["<eos>"]
        self.unk = self.ph_vocab["<unk>"]
        self.pad = self.ph_vocab["<pad>"]

    def _lstm_seq(self, w, r, b, x, h0, c0):
        H = self.H
        h, c = h0.copy(), c0.copy()
        ys = []
        bi, br = b[:4 * H], b[4 * H:]
        for t in range(x.shape[0]):
            g = w @ x[t] + r @ h + bi + br
            i = 1 / (1 + np.exp(-g[:H]))
            o = 1 / (1 + np.exp(-g[H:2 * H]))
            f = 1 / (1 + np.exp(-g[2 * H:3 * H]))
            cc = np.tanh(g[3 * H:])
            c = f * c + i * cc
            h = o * np.tanh(c)
            ys.append(h)
        return np.array(ys, dtype=np.float32), h, c

    def encode(self, word):
        H = self.H
        ids = [self.bos] + [self.char_vocab.get(ch, self.char_vocab.get("<unk>", 0))
                            for ch in word] + [self.eos]
        T = len(ids)
        x = self.W["encoder.embed.weight"][ids]                    # [T, E]
        hT, cT = {}, {}
        y = x
        for layer in (0, 1):
            out = np.zeros((T, 2 * H), dtype=np.float32)
            for d, rev in ((0, False), (1, True)):
                w = self.W[f"encoder.lstm.{layer}.weight"][d]       # [4H, in]
                r = self.W[f"encoder.lstm.{layer}.recurrence"][d]   # [4H, H]
                b = self.W[f"encoder.lstm.{layer}.bias"][d]         # [8H]
                xs = x if layer == 0 else y
                ys, hT[(layer, d)], cT[(layer, d)] = self._lstm_seq(
                    w, r, b, xs[::-1] if rev else xs,
                    np.zeros(H, np.float32), np.zeros(H, np.float32))
                if rev:
                    ys = ys[::-1]
                out[:, d * H:(d + 1) * H] = ys
            y = out
        enc = y @ self.W["encoder.projection.weight"].T + self.W["encoder.projection.bias"]
        h0 = np.stack([(hT[(0, 0)] + hT[(0, 1)]) / 2,
                       (hT[(1, 0)] + hT[(1, 1)]) / 2]).astype(np.float32)
        c0 = np.stack([(cT[(0, 0)] + cT[(0, 1)]) / 2,
                       (cT[(1, 0)] + cT[(1, 1)]) / 2]).astype(np.float32)
        return enc.astype(np.float32), h0, c0

    def decode_step(self, tok, h, c, enc):
        H = self.H
        q = h[1]                                                    # top layer
        kv_ = np.concatenate([np.repeat(q[None, :], enc.shape[0], 0), enc], 1)
        k = np.tanh(kv_ @ self.W["decoder.attention.key.weight"].T
                    + self.W["decoder.attention.key.bias"])
        sc = k @ self.W["decoder.attention.value.weight"].reshape(-1)
        a = np.exp(sc - sc.max())
        a /= a.sum()
        ctx_ = a @ enc
        emb = self.W["decoder.embed.weight"][tok]
        x0 = np.concatenate([emb, ctx_]).astype(np.float32)
        b0 = self.W["decoder.lstm.0.bias"].reshape(-1)
        g = (self.W["decoder.lstm.0.weight"] @ x0
             + self.W["decoder.lstm.0.recurrence"] @ h[0] + b0[:4 * H] + b0[4 * H:])
        i_ = 1 / (1 + np.exp(-g[:H]))
        o_ = 1 / (1 + np.exp(-g[H:2 * H]))
        f_ = 1 / (1 + np.exp(-g[2 * H:3 * H]))
        cc_ = np.tanh(g[3 * H:])
        c0n = f_ * c[0] + i_ * cc_
        h0n = o_ * np.tanh(c0n)
        b1 = self.W["decoder.lstm.1.bias"].reshape(-1)
        g = (self.W["decoder.lstm.1.weight"] @ h0n
             + self.W["decoder.lstm.1.recurrence"] @ h[1] + b1[:4 * H] + b1[4 * H:])
        i_ = 1 / (1 + np.exp(-g[:H]))
        o_ = 1 / (1 + np.exp(-g[H:2 * H]))
        f_ = 1 / (1 + np.exp(-g[2 * H:3 * H]))
        cc_ = np.tanh(g[3 * H:])
        c1n = f_ * c[1] + i_ * cc_
        h1n = o_ * np.tanh(c1n)
        fc_in = np.concatenate([h1n, ctx_]).astype(np.float32)       # [h1; context]
        fc = np.maximum(self.W["decoder.fc.0.weight"] @ fc_in
                        + self.W["decoder.fc.0.bias"], 0)
        logits = (self.W["decoder.fc.3.weight"] @ fc
                  + self.W["decoder.fc.3.bias"]).astype(np.float32)
        return logits, np.stack([h0n, h1n]).astype(np.float32), \
            np.stack([c0n, c1n]).astype(np.float32)

    def predict(self, word, beam_size=16, max_len=48):
        class Beam:
            def __init__(s, tokens, score, h, c, fin=False):
                s.tokens, s.score, s.h, s.c, s.fin = tokens, score, h, c, fin
            @property
            def ns(s):
                return s.score / max(1, len(s.tokens) - 1)

        enc, h0, c0 = self.encode(word.lower().strip())
        beams = [Beam([self.bos], 0.0, h0, c0)]
        for _ in range(max_len):
            if all(b.fin for b in beams):
                break
            cands = []
            for b in beams:
                if b.fin:
                    cands.append(b)
                    continue
                logits, nh, nc = self.decode_step(b.tokens[-1], b.h, b.c, enc)
                lp = logits.astype(np.float64)
                lp -= lp.max()
                lp -= np.log(np.exp(lp).sum())
                for pid in np.argsort(-lp, kind="stable")[:beam_size]:
                    pid = int(pid)
                    cands.append(Beam(b.tokens + [pid], b.score + float(lp[pid]),
                                      nh, nc, pid == self.eos))
            cands.sort(key=lambda b: b.ns, reverse=True)
            beams = cands[:beam_size]

        idx2ph = {v: k for k, v in self.ph_vocab.items()}
        out = []
        for b in beams:
            pron = [idx2ph[i] for i in b.tokens[1:]
                    if i not in (self.unk, self.pad, self.bos, self.eos) and i in idx2ph]
            if pron not in out:
                out.append(pron)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path,
                    default=Path(__file__).parent / "../../models/lstm-g2p-eng.gguf")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path(__file__).parent / "../tests/golden/lstm_g2p_eng_f32.json")
    args = ap.parse_args()

    g2p = NumpyLstmG2p(args.gguf)
    words = [
        "hello", "alignment", "phoneme", "tifa", "ggml", "xylem",
        "nhk", "helloworld", "separate", "different", "ka", "test",
        "yandex", "wohenhao", "openvpi", "diffsinger", "kakaru", " alignment",
    ]
    result = {}
    for w in words:
        result[w] = g2p.predict(w)
        print(w, "->", [" ".join(p) for p in result[w][:3]])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"wrote {args.output} ({len(result)} words)")


if __name__ == "__main__":
    main()
