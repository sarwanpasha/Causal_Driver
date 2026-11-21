"""Build cached, aligned matrices for the DepMap / TCGA experiments.

Panel selection depends only on DepMap (expression variance among genes that have
both expression and a Chronos gene-effect column), so the cross-library experiment
is not hostage to TCGA gene-name matching. TCGA is then aligned onto that panel by
gene SYMBOL (its row index is symbols, including LOC-style numeric symbols), and an
`in_tcga` mask records which panel genes were found.
"""
import argparse, os, re
import numpy as np, pandas as pd

GENE_RE = re.compile(r"^(.*?)\s*\((\d+)\)$")


def split_gene(col):
    m = GENE_RE.match(str(col).strip())
    return (m.group(1), int(m.group(2))) if m else (str(col).strip(), None)


def load_depmap_matrix(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str)
    return df


def first_by_symbol(cols):
    out = {}
    for c in cols:
        s, _ = split_gene(c)
        if s and s not in out:
            out[s] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tcga", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--panel", default="exprvar",
                    choices=["exprvar", "depvar", "depmix"],
                    help="exprvar: top-k by expression variance (what the original "
                         "pipeline used; it selects tissue-specific genes that carry "
                         "almost no fitness signal). depvar: top-k by variance of the "
                         "Chronos gene effect across cell lines, i.e. SELECTIVE "
                         "dependencies, which is what a dependency-prediction method "
                         "should actually be evaluated on. depmix: half of each.")
    ap.add_argument("--keep_y", action="store_true",
                    help="keep Y-chromosome genes (excluded by default: they encode "
                         "donor sex and act as a confounder in both domains)")
    a = ap.parse_args()

    print("loading DepMap expression ...", flush=True)
    expr = load_depmap_matrix(os.path.join(a.data, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"))
    print("  expr", expr.shape, flush=True)
    print("loading CRISPR gene effect ...", flush=True)
    ge = load_depmap_matrix(os.path.join(a.data, "CRISPRGeneEffect.csv"))
    print("  ge", ge.shape, flush=True)

    expr_sym = first_by_symbol(expr.columns)
    ge_sym = first_by_symbol(ge.columns)
    Y_CHROM = {"RPS4Y1", "RPS4Y2", "DDX3Y", "USP9Y", "UTY", "EIF1AY", "KDM5D", "ZFY",
               "NLGN4Y", "AMELY", "TMSB4Y", "TBL1Y", "PRKY", "TGIF2LY", "TXLNGY", "SRY",
               "PCDH11Y", "TSPY1", "TSPY2", "TSPY3", "TSPY4", "TSPY8", "TSPY10", "XIST",
               "DAZ1", "DAZ2", "DAZ3", "DAZ4", "CDY1", "CDY1B", "CDY2A", "CDY2B",
               "BPY2", "BPY2B", "BPY2C", "PRY", "PRY2", "VCY", "VCY1B", "XKRY", "XKRY2",
               "HSFY1", "HSFY2"}
    shared = sorted(set(expr_sym) & set(ge_sym))
    if not a.keep_y:
        before = len(shared)
        shared = [s for s in shared if s not in Y_CHROM]
        print(f"  sex-chromosome filter removed {before - len(shared)}", flush=True)
    print(f"  genes with both expression and gene effect: {len(shared)}", flush=True)

    ev = expr[[expr_sym[s] for s in shared]].var(axis=0).to_numpy()
    common_lines = sorted(set(expr.index) & set(ge.index))
    dv = ge.loc[common_lines, [ge_sym[s] for s in shared]].std(axis=0).to_numpy()
    dv = np.nan_to_num(dv)
    if a.panel == "exprvar":
        pick = np.argsort(-ev)[: a.top_k]
    elif a.panel == "depvar":
        pick = np.argsort(-dv)[: a.top_k]
    else:
        half = a.top_k // 2
        pe = list(np.argsort(-ev)[:half])
        pd_ = [i for i in np.argsort(-dv) if i not in set(pe)][: a.top_k - half]
        pick = np.array(pe + list(pd_))
    panel = [shared[i] for i in pick]
    print(f"  panel mode={a.panel}: dep-effect sd over panel "
          f"mean {dv[pick].mean():.4f} (all-gene mean {dv.mean():.4f})", flush=True)
    entrez = np.array([split_gene(expr_sym[s])[1] or -1 for s in panel])
    print(f"  panel {len(panel)}; e.g. {panel[:8]}", flush=True)

    lines = sorted(set(expr.index) & set(ge.index))
    X_s = expr.loc[lines, [expr_sym[s] for s in panel]].to_numpy(np.float32)
    D_s = ge.loc[lines, [ge_sym[s] for s in panel]].to_numpy(np.float32)
    print(f"  cell lines with both: {len(lines)}", flush=True)

    print("loading TCGA ...", flush=True)
    tc = pd.read_csv(a.tcga, sep="\t", index_col=0, low_memory=False)
    tc = tc[~tc.index.duplicated(keep="first")]
    tc.index = tc.index.astype(str)
    have = tc.index.intersection(pd.Index(panel))
    in_tcga = np.array([s in set(have) for s in panel])
    print(f"  tcga {tc.shape}; panel genes found in TCGA: {in_tcga.sum()}/{len(panel)}", flush=True)
    X_t = np.full((tc.shape[1], len(panel)), np.nan, dtype=np.float32)
    sub = tc.loc[[s for s in panel if s in set(have)]].to_numpy(np.float32).T
    X_t[:, np.flatnonzero(in_tcga)] = sub
    print(f"  tumours {X_t.shape}", flush=True)

    md = pd.read_csv(os.path.join(a.data, "Model.csv")).set_index("ModelID")["OncotreeLineage"]
    lineage = np.array([str(md.get(m, "Unknown")) for m in lines])

    print("loading ScreenGeneEffect ...", flush=True)
    sge = load_depmap_matrix(os.path.join(a.data, "ScreenGeneEffect.csv"))
    sge_sym = first_by_symbol(sge.columns)
    ssm = pd.read_csv(os.path.join(a.data, "ScreenSequenceMap.csv"))
    lib = ssm.dropna(subset=["ScreenID"]).groupby("ScreenID")["Library"].agg(
        lambda s: s.value_counts().index[0])
    smap = pd.read_csv(os.path.join(a.data, "CRISPRScreenMap.csv")).set_index("ScreenID")["ModelID"]
    present = [s for s in panel if s in sge_sym]
    print(f"  panel genes in ScreenGeneEffect: {len(present)}/{len(panel)}", flush=True)
    sub = sge[[sge_sym[s] for s in present]]
    sub = sub.reindex(columns=[sge_sym.get(s) for s in panel])
    sub.columns = panel
    info = pd.DataFrame({"Library": lib.reindex(sub.index), "ModelID": smap.reindex(sub.index)})
    out_lib = {}
    for name, tag in (("avana", "Avana"), ("ky", "KY")):
        idx = info.index[(info["Library"] == tag) & info["ModelID"].notna()]
        blk = sub.loc[idx].copy()
        blk["ModelID"] = info.loc[idx, "ModelID"].values
        agg = blk.groupby("ModelID").mean()
        out_lib[name + "_models"] = np.array(agg.index, dtype=object)
        out_lib["D_" + name] = agg.to_numpy(np.float32)
        print(f"  {tag}: {agg.shape[0]} models", flush=True)

    def ctrl(fn):
        s = pd.read_csv(os.path.join(a.data, fn))["Gene"].map(lambda c: split_gene(c)[0])
        return set(s.dropna())
    ce, ne = ctrl("AchillesCommonEssentialControls.csv"), ctrl("AchillesNonessentialControls.csv")
    common_essential = np.array([s in ce for s in panel])
    nonessential = np.array([s in ne for s in panel])
    print(f"  panel common-essential {common_essential.sum()}, nonessential {nonessential.sum()}",
          flush=True)

    np.savez_compressed(
        a.out, X_s=X_s, D_s=D_s, X_t=X_t, in_tcga=in_tcga,
        lines=np.array(lines, dtype=object), tumors=np.array(list(tc.columns), dtype=object),
        lineage=lineage, symbols=np.array(panel, dtype=object), entrez=entrez,
        common_essential=common_essential, nonessential=nonessential, **out_lib)
    print("wrote", a.out, os.path.getsize(a.out) / 1e6, "MB", flush=True)


if __name__ == "__main__":
    main()
