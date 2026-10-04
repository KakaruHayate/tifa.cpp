# Third-party notices

This project is MPL-2.0 (see [LICENSE](LICENSE)).  It depends on the following
third-party software; each keeps its own license.

## MeCab

The `japanese-mecab` G2P converter compiles the MeCab tokenizer subset from
[taku910/mecab](https://github.com/taku910/mecab) (pinned by
`cmake/Mecab.cmake`, portability fixes in `cmake/patches/mecab-portability.patch`).
MeCab is triple-licensed; we use the BSD option:

> Copyright (c) 2001-2008, Taku Kudo
> Copyright (c) 2004-2008, Nippon Telegraph and Telephone Corporation
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the names of the copyright owners nor the names of its
>   contributors may be used to endorse or promote products derived from this
>   software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## UniDic (unidic-lite)

The compiled dictionary shipped as the `unidic-lite-dicdir` release asset is
[unidic-mecab](https://clrd.ninjal.ac.jp/unidic/) packaged by the
[unidic-lite](https://pypi.org/project/unidic-lite/) project.  UniDic is
copyrighted free software by the UniDic Consortium, released under any of the
GPL, the LGPL, or the BSD license; the asset carries the original `COPYING`,
`GPL`, `LGPL` and `BSD` files:

> Copyright (c) 2011-2013, The UniDic Consortium
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> ... (see the `BSD` file inside the dictionary archive)

## Also bundled

* [pocketfft](https://github.com/mreineck/pocketfft) (BSD-3-Clause) — STFT for
  the mel frontend.
* [dr_libs](https://github.com/mackron/dr_libs) (Public Domain / MIT-0) — WAV
  reading in the CLI.
* [ggml](https://github.com/ggerganov/ggml) (MIT) — tensor engine, fetched
  pinned and patched (see `cmake/patches/`).
