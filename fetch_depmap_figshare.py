#!/usr/bin/env python3
"""Fetch DepMap Public 24Q4 files from the official figshare+ mirror (article 27993248).
The depmap.org portal API is now behind a human-verification wall; figshare is the
sanctioned bulk-download route. Run on a LOGIN NODE (compute nodes have no internet)."""
import argparse, hashlib, json, os, sys, urllib.request

ARTICLE = 27993248  # DepMap 24Q4 Public
WANT = [
    "CRISPRGeneEffect.csv",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "Model.csv",
    "ScreenGeneEffect.csv",
    "ScreenSequenceMap.csv",
    "CRISPRScreenMap.csv",
    "AchillesCommonEssentialControls.csv",
    "AchillesNonessentialControls.csv",
    "OmicsCNGene.csv",
    "Gene.csv",
]

def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--article", type=int, default=ARTICLE)
    ap.add_argument("--files", nargs="+", default=WANT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)
    meta = json.load(urllib.request.urlopen(
        f"https://api.figshare.com/v2/articles/{a.article}", timeout=120))
    print(f"article: {meta['title']}  ({len(meta['files'])} files)")
    by_name = {f["name"]: f for f in meta["files"]}
    missing = [n for n in a.files if n not in by_name]
    if missing:
        sys.stderr.write(f"NOT IN ARTICLE: {missing}\n")
    for name in a.files:
        if name not in by_name:
            continue
        f = by_name[name]
        dest = os.path.join(a.data, name)
        if os.path.exists(dest) and not a.force:
            if md5sum(dest) == f["computed_md5"]:
                print(f"  ok (cached)  {name}")
                continue
            print(f"  md5 mismatch, re-downloading {name}")
        print(f"  downloading  {name}  ({f['size']/1e6:.1f} MB)", flush=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(f["download_url"], tmp)
        got = md5sum(tmp)
        if got != f["computed_md5"]:
            sys.exit(f"MD5 MISMATCH for {name}: {got} != {f['computed_md5']}")
        os.replace(tmp, dest)
        print(f"    md5 ok  {got}", flush=True)
    print("done")

if __name__ == "__main__":
    main()
