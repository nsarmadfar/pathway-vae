"""Download the public data used in the PBMC experiment.

* PBMC 3k (10x Genomics), the processed AnnData used in the scanpy / cellxgene
  tutorials, including the standard cell-type annotation in ``obs['louvain']``.
* MSigDB Hallmark gene sets (Liberzon et al., 2015), v7.0, gene symbols.
  MSigDB gene sets are subject to the MSigDB license: https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp
  so they are downloaded rather than redistributed in this repository.
"""

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    ROOT / "data" / "raw" / "pbmc3k.h5ad":
        "https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad",
    ROOT / "data" / "gene_sets" / "h.all.v7.0.symbols.gmt":
        "https://raw.githubusercontent.com/zqfang/GSEApy/master/tests/extdata/h.all.v7.0.symbols.gmt",
}

if __name__ == "__main__":
    for path, url in FILES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"exists  {path.relative_to(ROOT)}")
            continue
        print(f"get     {url}")
        urlretrieve(url, path)
        print(f"saved   {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
