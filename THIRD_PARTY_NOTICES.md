# Third-party notices

This project is MPL-2.0 (see [LICENSE](LICENSE)).  It depends on the following
third-party software; each keeps its own license.

## Bundled / fetched

* [pocketfft](https://github.com/mreineck/pocketfft) (BSD-3-Clause) — STFT for
  the mel frontend.
* [dr_libs](https://github.com/mackron/dr_libs) (Public Domain / MIT-0) — WAV
  reading in the CLI.
* [ggml](https://github.com/ggerganov/ggml) (MIT) — tensor engine, fetched
  pinned and patched (see `cmake/patches/`).

## Model weights

TIFA-1.0-ST weights are distributed by the upstream
[openvpi/TIFA](https://github.com/openvpi/TIFA) project and keep their own
license; they are downloaded during conversion, not vendored here.

> G2P dictionaries, MeCab, UniDic and the BreathLab/FoxBreatheLabeler models
> were removed in 0.2.0 together with the G2P/breath subsystems.  The old
> notices remain in the `archive/main-v0.1.6` branch.
