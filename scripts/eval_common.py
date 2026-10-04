"""Benchmark harness for breath/AP/SP detectors against the hanser dataset.

Ground truth comes from each dataset TextGrid's `phones` tier: intervals
labelled AP or SP are the positives, everything else is lexical (V).  The
dataset has no EP labels, so EP is reported descriptively only.

Metrics
  frame  : per-frame label at the model's frame rate, resampled to 100 fps
           -> precision / recall / F1 per label
  event  : interval matching with an onset collar (default 200 ms, the same
           collar the BreathLab models are evaluated with)
"""
import json
import pathlib
import re

import numpy as np

FPS = 100.0


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

def read_textgrid_intervals(path, tier="phones"):
    """Return [(start, end, text)] for one tier of a TextGrid (Praat long/short)."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    # Split the file into tiers; keep the one whose name matches.
    items = re.split(r"\n\s*item \[\d+\]:", text)
    for item in items:
        m = re.search(r'name = "([^"]*)"', item)
        if not m or m.group(1) != tier:
            continue
        out = []
        for blk in re.finditer(
                r'xmin = ([-\d.eE]+)\s*\n\s*xmax = ([-\d.eE]+)\s*\n\s*text = "(.*?)"',
                item, re.S):
            out.append((float(blk.group(1)), float(blk.group(2)), blk.group(3)))
        return out
    return []


def truth_intervals(tg_path, tier="phones"):
    """[(start, end, label)] with label in {AP, SP, V}."""
    out = []
    for a, b, lab in read_textgrid_intervals(tg_path, tier):
        if b <= a:
            continue
        lab = lab.strip()
        if lab == "AP":
            out.append((a, b, "AP"))
        elif lab in ("SP", "sil", "pau", ""):
            out.append((a, b, "SP"))
        else:
            out.append((a, b, "V"))
    return out


def to_frames(intervals, n_frames, fps=FPS):
    """Rasterise [(start, end, label)] to a per-frame label array of length n."""
    lab = ["V"] * n_frames
    for a, b, l in intervals:
        i0 = max(0, int(round(a * fps)))
        i1 = min(n_frames, int(round(b * fps)))
        for i in range(i0, i1):
            lab[i] = l
    return lab


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def frame_scores(ref, hyp, labels=("AP", "SP", "EP")):
    """Per-label precision/recall/F1 over two equal-length label lists."""
    out = {}
    n = min(len(ref), len(hyp))
    for L in labels:
        tp = fp = fn = 0
        for i in range(n):
            r, h = ref[i] == L, hyp[i] == L
            if r and h:
                tp += 1
            elif h:
                fp += 1
            elif r:
                fn += 1
        out[L] = _prf(tp, fp, fn)
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f}


def events_of(labels, fps=FPS, want=("AP", "SP", "EP")):
    """Runs of one label in a frame array -> [(label, start, end)]."""
    out = []
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if labels[i] in want:
            out.append((labels[i], i / fps, j / fps))
        i = j
    return out


def event_scores(ref, hyp, collar=0.2, labels=("AP", "SP", "EP")):
    """Onset-collar event matching: a hypothesis matches a reference event of the
    same label when |onset difference| <= collar; each reference is used once."""
    out = {}
    for L in labels:
        R = [e for e in ref if e[0] == L]
        H = [e for e in hyp if e[0] == L]
        used = [False] * len(R)
        tp = 0
        for _, hs, he in sorted(H, key=lambda e: e[1]):
            best, bestd = -1, collar + 1e-9
            for i, (_, rs, re_) in enumerate(R):
                if used[i]:
                    continue
                d = abs(hs - rs)
                if d <= bestd:
                    best, bestd = i, d
            if best >= 0:
                used[best] = True
                tp += 1
        out[L] = _prf(tp, len(H) - tp, len(R) - tp)
    return out


def merge_scores(acc, new):
    for L, v in new.items():
        a = acc.setdefault(L, {"tp": 0, "fp": 0, "fn": 0})
        for k in ("tp", "fp", "fn"):
            a[k] += v[k]
    for L, a in acc.items():
        a.update({k: v for k, v in _prf(a["tp"], a["fp"], a["fn"]).items()
                  if k in ("p", "r", "f1")})
    return acc


def fmt(scores):
    parts = []
    for L in ("AP", "SP", "EP"):
        s = scores.get(L)
        if not s or s["tp"] + s["fp"] + s["fn"] == 0:
            continue
        parts.append(f"{L} P={s['p']:.3f} R={s['r']:.3f} F1={s['f1']:.3f} "
                     f"(tp{s['tp']}/fp{s['fp']}/fn{s['fn']})")
    return "  ".join(parts)


def dump(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


# --------------------------------------------------------------------------- #
# Energy-rule silence (the fallback our C++ engine uses for models with no SP
# head): a frame is SP when both its energy and its high-band energy sit near
# the noise floor, which keeps fricatives out.
# --------------------------------------------------------------------------- #

def frame_energy_db(y, sr, hop, frame_len, hp_hz=4000.0):
    """Per-frame broadband and high-band level in dB."""
    n = (len(y) - frame_len) // hop + 1
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    # one-pole high-pass for the guard band
    from scipy.signal import butter, lfilter
    b, a = butter(4, hp_hz / (sr / 2.0), btype="high")
    hi = lfilter(b, a, y).astype(np.float32)
    win = np.hanning(frame_len).astype(np.float32)
    idx = np.arange(frame_len)[None, :] + (np.arange(n) * hop)[:, None]
    e = np.sqrt((y[idx] ** 2 * win).mean(axis=1)) + 1e-9
    h = np.sqrt((hi[idx] ** 2 * win).mean(axis=1)) + 1e-9
    return (20 * np.log10(e)).astype(np.float32), (20 * np.log10(h)).astype(np.float32)


def energy_sp(energy_db, hf_db, fps, floor_pct=5.0, margin_db=8.0, hf_guard_db=6.0,
              min_dur_ms=120.0):
    """Frames that count as silence under the energy + high-band rule."""
    n = len(energy_db)
    sp = np.zeros(n, dtype=bool)
    if n == 0:
        return sp
    lim_e = np.percentile(energy_db, floor_pct) + margin_db
    lim_h = np.percentile(hf_db, floor_pct) + hf_guard_db
    sp = (energy_db <= lim_e) & (hf_db <= lim_h)
    minf = int(round(min_dur_ms * 0.001 * fps))
    i = 0
    while i < n:
        j = i
        while j < n and sp[j] == sp[i]:
            j += 1
        if sp[i] and (j - i) < minf:
            sp[i:j] = False
        i = j
    return sp
