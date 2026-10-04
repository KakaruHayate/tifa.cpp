"""Package the unidic-lite dicdir as the `unidic-lite-dicdir` release asset.

The `japanese-mecab` G2P converter needs a compiled MeCab/UniDic dictionary
(~260 MB extracted -- far too big for the per-platform bundles).  It ships as
a separate release asset: this script extracts `unidic_lite/dicdir` from the
PyPI sdist and re-zips it with a `unidic/` prefix, so users unpack one archive
next to the model:

    models/unidic/{sys.dic, matrix.bin, char.bin, unk.dic, mecabrc, ...}

The dictionary's own license files (BSD/GPL/LGPL/COPYING) live inside the
dicdir and are carried over.
"""
import argparse
import pathlib
import tarfile
import zipfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdist", required=True, help="the unidic_lite-<ver>.tar.gz from PyPI")
    ap.add_argument("--out", required=True, help="where to write the release zip")
    args = ap.parse_args()

    with tarfile.open(args.sdist, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "/dicdir/" in m.name and m.isfile()]
        if not members:
            raise SystemExit(f"no dicdir members found in {args.sdist}")
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zipped:
            for member in members:
                name = member.name.rsplit("/dicdir/", 1)[1]
                zipped.writestr(f"unidic/{name}", tar.extractfile(member).read())

    size_mb = pathlib.Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {len(members)} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
