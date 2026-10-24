import gc
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sweep

ROOT = Path("/home/rasert/workspace/Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao")
DADOS = ROOT / "dados"
CLASSES_VALIDAS = [1, 2, 3, 4, 5, 6, 7]
DIM = 600

def get_genes(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t", header=None)
    return df[0].astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True).tolist()

def get_valid_mask(path_labels: Path) -> np.ndarray:
    with open(path_labels, "r", encoding="utf-8") as f:
        rot = np.array([int(float(l.strip().split(",")[0])) for l in f if l.strip()], dtype=np.int64)
    return np.isin(rot, CLASSES_VALIDAS)

print("Carregando listas de genes...")
genes_pan = get_genes(DADOS / "pan" / "featuresPANcorrigido.tsv")
genes_fujita = get_genes(DADOS / "fujita" / "featuresF.tsv")
genes_mathys = get_genes(DADOS / "mathys" / "featuresM.tsv")

s_pan = set(genes_pan)
s_fujita = set(genes_fujita)
s_mathys = set(genes_mathys)
s_triple = s_pan & s_fujita & s_mathys
print(f"Total genes: Pan={len(genes_pan)}, Fujita={len(genes_fujita)}, Mathys={len(genes_mathys)}")
print(f"Interseção tripla: {len(s_triple)}")

orth_pan = np.load(DADOS / "pan" / "orthbase_pan_600d.npy")
orth_fujita = np.load(DADOS / "fujita" / "orthbase_fujita_600d.npy")
orth_mathys = np.load(DADOS / "mathys" / "orthbase_mathys_600d.npy")

# =========================================================================
# 1. MATHYS: Intra ruído, Alinhado a Fujita ruído, Alinhado a Pan ruído
# =========================================================================
p_mathys_intra = DADOS / "mathys" / "X_mathys_ruido_600d.npy"
p_mathys_fujita = DADOS / "mathys" / "X_mathys_alinhado_fujita_ruido_600d.npy"
p_mathys_pan = DADOS / "mathys" / "X_mathys_alinhado_pan_ruido_600d.npy"

if not (p_mathys_intra.exists() and p_mathys_fujita.exists() and p_mathys_pan.exists()):
    print("\n--- Processando Mathys (intra e alinhamentos) ---")
    mask_mathys = get_valid_mask(DADOS / "mathys" / "tipos_celulares_numericoMs.txt")
    adata = ad.read_h5ad(DADOS / "mathys" / "matriz_anotada_finalM.h5ad")
    X_sub = adata.X[mask_mathys]
    del adata; gc.collect()
    X_mathys_bin = X_sub.tocsr().astype(np.float32)
    del X_sub; gc.collect()
    X_mathys_bin.eliminate_zeros()
    X_mathys_bin.data = np.ones_like(X_mathys_bin.data, dtype=np.float32)
    n_cel = X_mathys_bin.shape[0]

    # 1.1 Mathys Intra Ruído
    if not p_mathys_intra.exists():
        t0 = time.time()
        mask_ruido_m = np.array([g not in s_triple for g in genes_mathys])
        print(f"Mathys Intra: {mask_ruido_m.sum()} colunas com 0.5")
        partes = []
        chunk = 3000
        for ini in range(0, n_cel, chunk):
            fim = min(ini + chunk, n_cel)
            bloco = X_mathys_bin[ini:fim].toarray().astype(np.float32)
            bloco[:, mask_ruido_m] = 0.5
            res = sweep.sweep(input=bloco, orthbase=orth_mathys, device="cuda", dtype=np.float32, verbose=False)
            partes.append(res.proj.astype(np.float32))
            del bloco
        X_proj = np.vstack(partes)
        np.save(p_mathys_intra, X_proj)
        print(f"Salvo {p_mathys_intra}: {X_proj.shape} em {time.time()-t0:.2f}s")
        del partes, X_proj; gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    # 1.2 Mathys Alinhado a Fujita (Exp 02)
    if not p_mathys_fujita.exists():
        t0 = time.time()
        mapa_fujita = {eid: idx for idx, eid in enumerate(genes_fujita)}
        alvo_idx, ref_idx = [], []
        for col_m, eid in enumerate(genes_mathys):
            if eid in mapa_fujita:
                alvo_idx.append(col_m)
                ref_idx.append(mapa_fujita[eid])
        alvo_idx = np.array(alvo_idx, dtype=np.int64)
        ref_idx = np.array(ref_idx, dtype=np.int64)
        print(f"Mathys -> Fujita: {len(ref_idx)} em comum, {len(genes_fujita) - len(ref_idx)} ausentes (0.5)")
        
        X_comum = X_mathys_bin[:, alvo_idx].tocsr()
        partes = []
        chunk = 3000
        n_ref = len(genes_fujita)
        for ini in range(0, n_cel, chunk):
            fim = min(ini + chunk, n_cel)
            bloco = np.full((fim - ini, n_ref), 0.5, dtype=np.float32)
            bloco[:, ref_idx] = X_comum[ini:fim].toarray().astype(np.float32)
            res = sweep.sweep(input=bloco, orthbase=orth_fujita, device="cuda", dtype=np.float32, verbose=False)
            partes.append(res.proj.astype(np.float32))
            del bloco
        X_proj = np.vstack(partes)
        np.save(p_mathys_fujita, X_proj)
        print(f"Salvo {p_mathys_fujita}: {X_proj.shape} em {time.time()-t0:.2f}s")
        del partes, X_proj, X_comum; gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    # 1.3 Mathys Alinhado a Pan (Exp 06)
    if not p_mathys_pan.exists():
        t0 = time.time()
        mapa_pan = {eid: idx for idx, eid in enumerate(genes_pan)}
        alvo_idx, ref_idx = [], []
        for col_m, eid in enumerate(genes_mathys):
            if eid in mapa_pan:
                alvo_idx.append(col_m)
                ref_idx.append(mapa_pan[eid])
        alvo_idx = np.array(alvo_idx, dtype=np.int64)
        ref_idx = np.array(ref_idx, dtype=np.int64)
        print(f"Mathys -> Pan: {len(ref_idx)} em comum, {len(genes_pan) - len(ref_idx)} ausentes (0.5)")

        X_comum = X_mathys_bin[:, alvo_idx].tocsr()
        partes = []
        chunk = 2500
        n_ref = len(genes_pan)
        for ini in range(0, n_cel, chunk):
            fim = min(ini + chunk, n_cel)
            bloco = np.full((fim - ini, n_ref), 0.5, dtype=np.float32)
            bloco[:, ref_idx] = X_comum[ini:fim].toarray().astype(np.float32)
            res = sweep.sweep(input=bloco, orthbase=orth_pan, device="cuda", dtype=np.float32, verbose=False)
            partes.append(res.proj.astype(np.float32))
            del bloco
        X_proj = np.vstack(partes)
        np.save(p_mathys_pan, X_proj)
        print(f"Salvo {p_mathys_pan}: {X_proj.shape} em {time.time()-t0:.2f}s")
        del partes, X_proj, X_comum; gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    del X_mathys_bin; gc.collect()

# =========================================================================
# 2. FUJITA: Alinhado a Pan ruído (Exp 05)
# =========================================================================
p_fujita_pan = DADOS / "fujita" / "X_fujita_alinhado_pan_ruido_600d.npy"
if not p_fujita_pan.exists():
    print("\n--- Processando Fujita -> Pan (Exp 05) ---")
    mask_fujita = get_valid_mask(DADOS / "fujita" / "cell_types_binarioF.txt")
    adata = ad.read_h5ad(DADOS / "fujita" / "matrizFiltradaeNormalizadaF (1).h5ad")
    X_sub = adata.X[mask_fujita]
    del adata; gc.collect()
    X_fujita_bin = X_sub.tocsr().astype(np.float32)
    del X_sub; gc.collect()
    X_fujita_bin.eliminate_zeros()
    X_fujita_bin.data = np.ones_like(X_fujita_bin.data, dtype=np.float32)
    n_cel = X_fujita_bin.shape[0]

    t0 = time.time()
    mapa_pan = {eid: idx for idx, eid in enumerate(genes_pan)}
    alvo_idx, ref_idx = [], []
    for col_f, eid in enumerate(genes_fujita):
        if eid in mapa_pan:
            alvo_idx.append(col_f)
            ref_idx.append(mapa_pan[eid])
    alvo_idx = np.array(alvo_idx, dtype=np.int64)
    ref_idx = np.array(ref_idx, dtype=np.int64)
    print(f"Fujita -> Pan: {len(ref_idx)} em comum, {len(genes_pan) - len(ref_idx)} ausentes (0.5)")

    X_comum = X_fujita_bin[:, alvo_idx].tocsr()
    del X_fujita_bin; gc.collect()
    partes = []
    chunk = 2500
    n_ref = len(genes_pan)
    for ini in range(0, n_cel, chunk):
        fim = min(ini + chunk, n_cel)
        bloco = np.full((fim - ini, n_ref), 0.5, dtype=np.float32)
        bloco[:, ref_idx] = X_comum[ini:fim].toarray().astype(np.float32)
        res = sweep.sweep(input=bloco, orthbase=orth_pan, device="cuda", dtype=np.float32, verbose=False)
        partes.append(res.proj.astype(np.float32))
        del bloco
    X_proj = np.vstack(partes)
    np.save(p_fujita_pan, X_proj)
    print(f"Salvo {p_fujita_pan}: {X_proj.shape} em {time.time()-t0:.2f}s")
    del partes, X_proj, X_comum; gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

# =========================================================================
# 3. PAN: Intra ruído (Exp 04)
# =========================================================================
p_pan_intra = DADOS / "pan" / "X_pan_ruido_600d.npy"
if not p_pan_intra.exists():
    print("\n--- Processando Pan Intra Ruído (Exp 04) ---")
    mask_pan = get_valid_mask(DADOS / "pan" / "PanNumerico.csv")
    adata = ad.read_h5ad(DADOS / "pan" / "pan_anotado.h5ad")
    X_sub = adata.X[mask_pan]
    del adata; gc.collect()
    X_pan_bin = X_sub.tocsr().astype(np.float32)
    del X_sub; gc.collect()
    X_pan_bin.eliminate_zeros()
    X_pan_bin.data = np.ones_like(X_pan_bin.data, dtype=np.float32)
    n_cel = X_pan_bin.shape[0]

    t0 = time.time()
    mask_ruido_p = np.array([g not in s_triple for g in genes_pan])
    print(f"Pan Intra: {mask_ruido_p.sum()} colunas com 0.5 de {len(genes_pan)}")
    partes = []
    chunk = 2500
    for ini in range(0, n_cel, chunk):
        fim = min(ini + chunk, n_cel)
        bloco = X_pan_bin[ini:fim].toarray().astype(np.float32)
        bloco[:, mask_ruido_p] = 0.5
        res = sweep.sweep(input=bloco, orthbase=orth_pan, device="cuda", dtype=np.float32, verbose=False)
        partes.append(res.proj.astype(np.float32))
        del bloco
    X_proj = np.vstack(partes)
    np.save(p_pan_intra, X_proj)
    print(f"Salvo {p_pan_intra}: {X_proj.shape} em {time.time()-t0:.2f}s")
    del partes, X_proj, X_pan_bin; gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

print("\nTodas as projeções com ruído foram pré-computadas e validadas com sucesso!")
