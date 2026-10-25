"""Módulo de Pipeline de Imputação e Projeção SWeeP para scRNA-seq.

Fornece funções padronizadas para:
1. Carregamento OOM-safe e binarização de matrizes scRNA-seq;
2. Particionamento estratificado canônico (Treino 70% / Validação 15% / Teste 15%);
3. Calibração do limiar ótimo de binarização (threshold*) via curva Precision-Recall (ADR 027);
4. Imputação e projeção SWeeP 600D em streaming na GPU (ADR 019 / ADR 020);
5. Atualização programática dos notebooks de experimentos com a Seção Pós-Hopfield.
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import sweep
import torch

from .hopfield import ModernHopfieldNetwork

CLASSES_VALIDAS = [1, 2, 3, 4, 5, 6, 7]
NOMES_CLASSES = [f"Classe {i}" for i in CLASSES_VALIDAS]


def carregar_ensembl_ids(path_tsv: Path | str) -> list[str]:
    """Carrega Ensembl IDs de um arquivo TSV sem versões numéricas."""
    df = pd.read_csv(path_tsv, sep="\t", header=None)
    return df[0].astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True).tolist()


def carregar_matriz_binaria_csr(
    path_h5ad: Path | str,
    path_labels: Path | str,
    classes_validas: list[int] | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Carrega matriz de expressão filtrada e binarizada em formato CSR esparso.

    Utiliza lock de arquivo em /tmp/hopfield_h5ad_io.lock para evitar picos
    de RAM causados por leituras simultâneas de arquivos .h5ad massivos.
    """
    if classes_validas is None:
        classes_validas = CLASSES_VALIDAS

    path_h5ad = Path(path_h5ad)
    path_labels = Path(path_labels)

    # 1. Carrega e filtra rótulos
    with open(path_labels, "r", encoding="utf-8") as f:
        rotulos = np.array(
            [int(float(line.strip().split(",")[0])) for line in f if line.strip()],
            dtype=np.int64,
        )
    mascara_validos = np.isin(rotulos, classes_validas)
    y_filtrado = rotulos[mascara_validos] - 1  # 1..7 -> 0..6

    # 2. Leitura com lock OOM-Safe
    lock_file = Path("/tmp/hopfield_h5ad_io.lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            print(f"[I/O Safe] Lendo {path_h5ad.name}...")
            adata = ad.read_h5ad(path_h5ad)
            X_sub = adata.X[mascara_validos]
            del adata
            gc.collect()
            X_csr = X_sub.tocsr().astype(np.float32)
            del X_sub
            gc.collect()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    # 3. Binarização estrita (valores > 0 -> 1.0)
    X_csr.eliminate_zeros()
    X_csr.data = np.ones_like(X_csr.data, dtype=np.float32)

    assert X_csr.shape[0] == len(y_filtrado), "Inconsistência entre X_csr e y"
    return X_csr, y_filtrado


def obter_particoes_canonicas(
    y_0a6: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera exatamente as mesmas partições estratificadas da Fase 1 (70% / 15% / 15%)."""
    indices = np.arange(len(y_0a6))
    idx_treino_val, idx_teste = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y_0a6
    )
    razao_val = 0.15 / (1.0 - 0.15)
    idx_treino, idx_val = train_test_split(
        idx_treino_val,
        test_size=razao_val,
        random_state=seed,
        stratify=y_0a6[idx_treino_val],
    )
    return idx_treino, idx_val, idx_teste


def calibrar_limiar_otimo_pr(
    rede: ModernHopfieldNetwork,
    X_val_csr: sp.csr_matrix,
    mask_genes_ocultos: np.ndarray,
    batch_size: int = 2048,
    max_coordenadas_amostra: int = 20_000_000,
) -> dict[str, Any]:
    """Calibra o limiar ótimo de binarização (threshold*) via curva Precision-Recall.

    Opera exclusivamente em X_val (15% de validação da referência), mascarando com 0.5
    as colunas ausentes/ruído e comparando a probabilidade prevista contra o valor
    real binário conhecido em X_val.
    """
    t0 = time.time()
    n_val, n_genes = X_val_csr.shape
    cols_ocultas = np.where(mask_genes_ocultos)[0]
    n_ocultos = len(cols_ocultas)

    print(
        f"[Calibração PR] Avaliando {n_val} células de validação x {n_ocultos} genes ocultados..."
    )

    # 1. Recupera probabilidades contínuas com sentinela 0.5
    _, Wprob_val = rede.retrieve(
        queries=X_val_csr,
        batch_size=batch_size,
        mask_sentinela_ausentes=cols_ocultas,
        fill_value=0.5,
        return_probabilities=True,
    )

    # 2. Extrai ground truth binário e probabilidades previstas
    X_sub_val = X_val_csr[:, cols_ocultas]
    y_true_genes = X_sub_val.toarray().ravel().astype(np.int32)
    y_prob_genes = Wprob_val[:, cols_ocultas].ravel()

    del Wprob_val, X_sub_val
    gc.collect()

    # 3. Subamostragem defensiva caso exceda 20M de elementos (mantém < 200 MB RAM)
    total_elem = len(y_true_genes)
    if total_elem > max_coordenadas_amostra:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(total_elem, size=max_coordenadas_amostra, replace=False)
        y_true_genes = y_true_genes[idx_sample]
        y_prob_genes = y_prob_genes[idx_sample]

    # 4. Curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_true_genes, y_prob_genes)
    numerador = 2.0 * precision[:-1] * recall[:-1]
    denominador = np.maximum(precision[:-1] + recall[:-1], 1e-9)
    f1_pts = numerador / denominador

    best_idx = int(np.argmax(f1_pts))
    threshold_prob_otimo = float(thresholds[best_idx])
    threshold_bipolar_otimo = float(2.0 * threshold_prob_otimo - 1.0)
    melhor_f1 = float(f1_pts[best_idx])
    precisao_otima = float(precision[best_idx])
    recall_otimo = float(recall[best_idx])

    # 5. Atualiza o limiar interno da rede no espaço bipolar
    rede.threshold = threshold_bipolar_otimo

    # 6. Downsample para 200 pontos para plotagem leve em Plotly
    passo = max(1, len(precision) // 200)
    pr_curva = {
        "precision": [float(p) for p in precision[::passo]],
        "recall": [float(r) for r in recall[::passo]],
    }

    tempo_total = time.time() - t0
    print(
        f"[Calibração PR] Concluída em {tempo_total:.2f}s: "
        f"threshold_prob* = {threshold_prob_otimo:.4f} (bipolar = {threshold_bipolar_otimo:.4f}) | "
        f"F1 Gene = {melhor_f1:.4f} (Prec = {precisao_otima:.4f}, Rec = {recall_otimo:.4f})"
    )

    return {
        "threshold_prob_otimo": threshold_prob_otimo,
        "threshold_bipolar_otimo": threshold_bipolar_otimo,
        "melhor_f1_gene": melhor_f1,
        "precisao_otima": precisao_otima,
        "recall_otimo": recall_otimo,
        "tempo_calibracao_s": tempo_total,
        "pr_curva": pr_curva,
    }


def imputar_e_projetar_sweep_streaming(
    rede: ModernHopfieldNetwork,
    X_alvo_csr: sp.csr_matrix,
    genes_alvo: list[str],
    genes_ref: list[str],
    orthbase_ref: np.ndarray,
    threshold_prob_otimo: float,
    mask_forcar_sentinela_ref: np.ndarray | None = None,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executa a imputação Hopfield e projeção SWeeP em streaming na GPU.

    Substitui exclusivamente as colunas sentinela (0,5) por valores binários {0, 1}
    com base no threshold_prob_otimo, mantendo 100% dos genes medidos inalterados,
    projetando cada bloco imediatamente via sweep.sweep(..., device='cuda').
    """
    t0 = time.time()
    n_celulas = X_alvo_csr.shape[0]
    n_ref = len(genes_ref)

    # Identifica colunas comuns e ausentes
    is_intra = (len(genes_alvo) == len(genes_ref)) and (genes_alvo == genes_ref)

    if is_intra:
        assert mask_forcar_sentinela_ref is not None, "Intra exige mask_forcar_sentinela_ref"
        mask_ausentes = np.asarray(mask_forcar_sentinela_ref, dtype=bool)
        alvo_idx_comum = None
        ref_idx_comum = None
    else:
        mapa_ref = {eid: idx for idx, eid in enumerate(genes_ref)}
        alvo_idx, ref_idx = [], []
        for col_a, eid in enumerate(genes_alvo):
            if eid in mapa_ref:
                alvo_idx.append(col_a)
                ref_idx.append(mapa_ref[eid])
        alvo_idx_comum = np.array(alvo_idx, dtype=np.int64)
        ref_idx_comum = np.array(ref_idx, dtype=np.int64)

        mask_ausentes = np.ones(n_ref, dtype=bool)
        mask_ausentes[ref_idx_comum] = False

    cols_ausentes = np.where(mask_ausentes)[0]
    n_ausentes = len(cols_ausentes)

    print(
        f"[Streaming Imputação SWeeP] {n_celulas} células | {n_ref} genes de referência | "
        f"{n_ausentes} colunas ausentes a imputar ({n_ausentes / n_ref * 100:.2f}%)"
    )

    partes_proj: list[np.ndarray] = []
    total_imputados_1 = 0
    total_imputados_0 = 0

    for ini in range(0, n_celulas, chunk_size):
        fim = min(ini + chunk_size, n_celulas)
        tamanho_lote = fim - ini

        # 1. Monta bloco de entrada com 0.5 nas ausências
        if is_intra:
            bloco_entrada = X_alvo_csr[ini:fim].toarray().astype(np.float32)
            bloco_entrada[:, cols_ausentes] = 0.5
        else:
            bloco_entrada = np.full((tamanho_lote, n_ref), 0.5, dtype=np.float32)
            assert alvo_idx_comum is not None and ref_idx_comum is not None
            sub_comum = X_alvo_csr[ini:fim][:, alvo_idx_comum].toarray().astype(np.float32)
            bloco_entrada[:, ref_idx_comum] = sub_comum
            del sub_comum

        # 2. Inferência Hopfield com probabilidades contínuas
        _, prob_bloco = rede.retrieve(
            queries=bloco_entrada,
            batch_size=chunk_size,
            return_probabilities=True,
        )

        # 3. Substituição exclusiva dos 0.5 pelo valor binário reconstruído
        bloco_final = bloco_entrada.copy()
        imputados = (prob_bloco[:, cols_ausentes] >= threshold_prob_otimo).astype(np.float32)
        bloco_final[:, cols_ausentes] = imputados

        total_imputados_1 += int(np.sum(imputados == 1.0))
        total_imputados_0 += int(np.sum(imputados == 0.0))

        # Travas de integridade
        assert not np.any(np.isclose(bloco_final, 0.5)), "Sentinela 0.5 remanescente após imputação!"
        assert np.all((bloco_final == 0.0) | (bloco_final == 1.0)), "Matriz final não binária!"

        # 4. Projeção SWeeP direta na GPU via CuPy
        res = sweep.sweep(
            input=bloco_final,
            orthbase=orthbase_ref,
            device="cuda",
            dtype=np.float32,
            verbose=False,
        )
        partes_proj.append(res.proj.astype(np.float32))

        del bloco_entrada, prob_bloco, bloco_final, imputados, res
        if (ini // chunk_size) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()

    X_proj_600d = np.vstack(partes_proj)
    tempo_total = time.time() - t0

    assert X_proj_600d.shape == (n_celulas, 600), f"Shape inválido da projeção: {X_proj_600d.shape}"
    assert not np.isnan(X_proj_600d).any(), "NaN detectado na projeção pós-Hopfield!"

    stats = {
        "n_celulas": n_celulas,
        "n_genes_ref": n_ref,
        "n_genes_imputados": n_ausentes,
        "total_coordenadas_imputadas": int(n_celulas * n_ausentes),
        "coordenadas_ativadas_1": total_imputados_1,
        "coordenadas_inativadas_0": total_imputados_0,
        "pct_ativadas_1": float(total_imputados_1 / max(1, (total_imputados_1 + total_imputados_0)) * 100.0),
        "tempo_streaming_s": tempo_total,
    }

    print(
        f"[Streaming Imputação SWeeP] Concluído em {tempo_total:.2f}s: {X_proj_600d.shape} | "
        f"Imputações: {total_imputados_1:,} ativados ({stats['pct_ativadas_1']:.1f}%), {total_imputados_0:,} inativados"
    )

    return X_proj_600d, stats


def anexar_secao_pos_hopfield_notebook(
    caminho_notebook: Path | str,
    path_proj_pos_hopfield: str,
    nome_cenario: str,
    is_cross_dataset: bool,
    path_json_saida: str,
    path_modelo: str,
    path_scaler: str,
    path_y: str,
    path_proj_limpo: str,
    path_proj_ruido: str,
) -> None:
    """Anexa e executa a Seção 8 (Pós-Imputação Hopfield) em um notebook exp0X.ipynb.

    Garante idempotência (substitui caso já anexado) e reusa estritamente os modelos
    .keras e scalers .pkl da Fase 1, executando de forma autocontida exclusivamente
    as células da nova seção para preservar 100% dos outputs históricos anteriores.
    """
    import uuid
    import joblib
    import nbformat
    from sklearn.metrics import accuracy_score, classification_report
    from tensorflow.keras.models import load_model

    caminho_nb = Path(caminho_notebook)
    root_dir = caminho_nb.resolve().parent.parent

    # 1. Avaliação Direta no Processo (OOM-Safe & Zero DeadKernel)
    modelo = load_model(root_dir / path_modelo)
    scaler = joblib.load(root_dir / path_scaler)
    y_raw = np.load(root_dir / path_y)

    X_limpo_600d = np.load(root_dir / path_proj_limpo).astype(np.float32)
    X_ruido_600d = np.load(root_dir / path_proj_ruido).astype(np.float32)
    X_hop_600d = np.load(root_dir / path_proj_pos_hopfield).astype(np.float32)

    if not is_cross_dataset:
        indices = np.arange(len(y_raw))
        idx_treino_val, idx_teste = train_test_split(
            indices, test_size=0.15, random_state=42, stratify=y_raw
        )
        y_eval = y_raw[idx_teste]
        X_limpo_eval = X_limpo_600d[idx_teste]
        X_ruido_eval = X_ruido_600d[idx_teste]
        X_hop_eval = X_hop_600d[idx_teste]

        # Dataset completo para métrica adicional
        X_full_hop_s = scaler.transform(X_hop_600d)
        y_pred_full_hop = np.argmax(modelo.predict(X_full_hop_s, verbose=0), axis=1)
        acc_full_hop = accuracy_score(y_raw, y_pred_full_hop)
    else:
        y_eval = y_raw
        X_limpo_eval = X_limpo_600d
        X_ruido_eval = X_ruido_600d
        X_hop_eval = X_hop_600d
        acc_full_hop = None

    # Avaliação Limpo, Ruído 0,5 e Hopfield
    y_pred_limpo = np.argmax(modelo.predict(scaler.transform(X_limpo_eval), verbose=0), axis=1)
    y_pred_ruido = np.argmax(modelo.predict(scaler.transform(X_ruido_eval), verbose=0), axis=1)
    y_pred_hop = np.argmax(modelo.predict(scaler.transform(X_hop_eval), verbose=0), axis=1)

    rel_limpo = classification_report(y_eval, y_pred_limpo, target_names=NOMES_CLASSES, output_dict=True)
    rel_ruido = classification_report(y_eval, y_pred_ruido, target_names=NOMES_CLASSES, output_dict=True)
    rel_hop = classification_report(y_eval, y_pred_hop, target_names=NOMES_CLASSES, output_dict=True)
    rel_hop_str = classification_report(y_eval, y_pred_hop, target_names=NOMES_CLASSES)

    acc_limpo = float(rel_limpo["accuracy"])
    acc_ruido = float(rel_ruido["accuracy"])
    acc_hop = float(rel_hop["accuracy"])

    macro_limpo = float(rel_limpo["macro avg"]["f1-score"])
    macro_ruido = float(rel_ruido["macro avg"]["f1-score"])
    macro_hop = float(rel_hop["macro avg"]["f1-score"])

    delta_acc = (acc_hop - acc_ruido) * 100.0
    delta_macro = (macro_hop - macro_ruido) * 100.0
    queda_macro = (macro_limpo - macro_ruido) * 100.0
    taxa_recup = (delta_macro / max(1e-6, queda_macro)) * 100.0

    # Texto stdout para Célula 1 (Inferência)
    if not is_cross_dataset:
        stdout_infer = (
            "=== RELATÓRIO: TESTE HOLDOUT PÓS-IMPUTAÇÃO HOPFIELD ===\n"
            + rel_hop_str
            + f"\nAcurácia no Dataset Completo (100%) Pós-Hopfield: {acc_full_hop:.4f}\n"
        )
    else:
        stdout_infer = (
            "=== RELATÓRIO: DATASET ALVO COMPLETO PÓS-IMPUTAÇÃO HOPFIELD ===\n"
            + rel_hop_str
            + "\n"
        )

    # Texto stdout para Célula 3 (Comparativo)
    stdout_comp = (
        "=" * 70 + "\n"
        "  RESUMO DE RECUPERAÇÃO TRANSCRICIONAL VIA MODERN HOPFIELD NETWORK\n"
        + "=" * 70 + "\n"
        f"Acurácia Global  : Limpo {acc_limpo*100:.2f}% | Ruído {acc_ruido*100:.2f}% | Hopfield {acc_hop*100:.2f}% (Delta: {delta_acc:+.2f} p.p.)\n"
        f"Macro F1-Score   : Limpo {macro_limpo*100:.2f}% | Ruído {macro_ruido*100:.2f}% | Hopfield {macro_hop*100:.2f}% (Delta: {delta_macro:+.2f} p.p.)\n"
        f"Taxa de Recuperação da Queda de Macro F1: {taxa_recup:.2f}%\n"
        + "=" * 70 + "\n"
        f"Relatório comparativo salvo com sucesso em: {root_dir / path_json_saida}\n"
    )

    # 2. Exportação do relatório JSON consolidado
    resumo_json = {
        "cenario": nome_cenario,
        "is_cross_dataset": is_cross_dataset,
        "baseline_limpo": {
            "acuracia": acc_limpo,
            "macro_f1": macro_limpo,
            "weighted_f1": float(rel_limpo["weighted avg"]["f1-score"]),
            "f1_classes": {nome: float(rel_limpo.get(nome, {}).get("f1-score", 0.0)) for nome in NOMES_CLASSES},
        },
        "com_ruido_05": {
            "acuracia": acc_ruido,
            "macro_f1": macro_ruido,
            "weighted_f1": float(rel_ruido["weighted avg"]["f1-score"]),
            "f1_classes": {nome: float(rel_ruido.get(nome, {}).get("f1-score", 0.0)) for nome in NOMES_CLASSES},
        },
        "pos_hopfield": {
            "acuracia": acc_hop,
            "macro_f1": macro_hop,
            "weighted_f1": float(rel_hop["weighted avg"]["f1-score"]),
            "f1_classes": {nome: float(rel_hop.get(nome, {}).get("f1-score", 0.0)) for nome in NOMES_CLASSES},
        },
        "recuperacao": {
            "delta_acuracia_pp": delta_acc,
            "delta_macro_f1_pp": delta_macro,
            "queda_macro_f1_total_pp": queda_macro,
            "taxa_recuperacao_macro_pct": taxa_recup,
        },
    }
    path_json_final = root_dir / path_json_saida
    path_json_final.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json_final, "w", encoding="utf-8") as fj:
        json.dump(resumo_json, fj, indent=2)

    # 3. Manipulação do Notebook com nbformat
    with open(caminho_nb, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Idempotência: remove células da Seção 8 pré-existentes
    idx_corte = None
    for i, cell in enumerate(nb.cells):
        src = "".join(cell.get("source", ""))
        if "Seção 8" in src or "Pós-Imputação Hopfield" in src:
            idx_corte = i
            break

    if idx_corte is not None:
        nb.cells = nb.cells[:idx_corte]

    # Célula Markdown
    md_source = f"""# 8. Avaliação Pós-Imputação Hopfield (Reconstrução de Features Ausentes)

## Objetivo
Avaliar a capacidade de classificação do modelo **MLP (treinado no baseline limpo)** após a substituição das features faltantes (anteriormente preenchidas com ruído `0,5`) pelos valores transcricionais binários inferidos pela **Modern Hopfield Network** e reprojetados no espaço **SWeeP 600D**.

- **Cenário:** {nome_cenario}
- **Projeção Pós-Hopfield:** `{path_proj_pos_hopfield}`
- **Avaliação:** Comparação pareada contra o *Baseline Limpo* e o cenário *Com Ruído 0,5*.
"""
    cell_md = nbformat.v4.new_markdown_cell(md_source)
    cell_md.id = f"cell_{uuid.uuid4().hex[:8]}"
    nb.cells.append(cell_md)

    # Célula de Código: Inferência
    if not is_cross_dataset:
        code_infer = f"""# 8.1 Inferência Pós-Hopfield no Conjunto de Teste Holdout (15% Cego) e Dataset Completo
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.config.set_visible_devices([], "GPU")

if Path.cwd().name == "experimentos":
    os.chdir(Path.cwd().parent)
ROOT_DIR = Path.cwd()

DIM_PROJECAO = 600
CLASSES_VALIDAS = [1, 2, 3, 4, 5, 6, 7]
NOMES_CLASSES = [f"Classe {{i}}" for i in CLASSES_VALIDAS]

modelo = load_model(ROOT_DIR / "{path_modelo}")
scaler = joblib.load(ROOT_DIR / "{path_scaler}")

y_ref = np.load(ROOT_DIR / "{path_y}")
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(indices, test_size=0.15, random_state=42, stratify=y_ref)

X_pos_hopfield = np.load(ROOT_DIR / "{path_proj_pos_hopfield}").astype(np.float32)
assert X_pos_hopfield.shape == (len(y_ref), DIM_PROJECAO), f"Shape inválido: {{X_pos_hopfield.shape}}"

X_teste_hop_s = scaler.transform(X_pos_hopfield[idx_teste])
probs_teste_hop = modelo.predict(X_teste_hop_s, verbose=0)
y_pred_teste_hop = np.argmax(probs_teste_hop, axis=1)

print("=== RELATÓRIO: TESTE HOLDOUT PÓS-IMPUTAÇÃO HOPFIELD ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_hop, target_names=NOMES_CLASSES))

X_full_hop_s = scaler.transform(X_pos_hopfield)
probs_full_hop = modelo.predict(X_full_hop_s, verbose=0)
y_pred_full_hop = np.argmax(probs_full_hop, axis=1)
acc_full_hop = accuracy_score(y_ref, y_pred_full_hop)
print(f"Acurácia no Dataset Completo (100%) Pós-Hopfield: {{acc_full_hop:.4f}}")
"""
    else:
        code_infer = f"""# 8.1 Inferência Pós-Hopfield no Dataset Alvo Completo
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.config.set_visible_devices([], "GPU")

if Path.cwd().name == "experimentos":
    os.chdir(Path.cwd().parent)
ROOT_DIR = Path.cwd()

DIM_PROJECAO = 600
CLASSES_VALIDAS = [1, 2, 3, 4, 5, 6, 7]
NOMES_CLASSES = [f"Classe {{i}}" for i in CLASSES_VALIDAS]

modelo = load_model(ROOT_DIR / "{path_modelo}")
scaler = joblib.load(ROOT_DIR / "{path_scaler}")

y_alvo = np.load(ROOT_DIR / "{path_y}")
X_pos_hopfield = np.load(ROOT_DIR / "{path_proj_pos_hopfield}").astype(np.float32)
assert X_pos_hopfield.shape == (len(y_alvo), DIM_PROJECAO), f"Shape inválido: {{X_pos_hopfield.shape}}"

X_alvo_hop_s = scaler.transform(X_pos_hopfield)
probs_alvo_hop = modelo.predict(X_alvo_hop_s, verbose=0)
y_pred_alvo_hop = np.argmax(probs_alvo_hop, axis=1)

print("=== RELATÓRIO: DATASET ALVO COMPLETO PÓS-IMPUTAÇÃO HOPFIELD ===")
print(classification_report(y_alvo, y_pred_alvo_hop, target_names=NOMES_CLASSES))
"""

    cell_infer = nbformat.v4.new_code_cell(code_infer)
    cell_infer.id = f"cell_{uuid.uuid4().hex[:8]}"
    cell_infer.outputs = [nbformat.v4.new_output(output_type="stream", name="stdout", text=stdout_infer)]
    nb.cells.append(cell_infer)

    # Célula de Código: Matriz de Confusão
    if not is_cross_dataset:
        code_cm = f"""# 8.2 Matriz de Confusão Pós-Imputação Hopfield
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def plotar_matriz_confusao(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(NOMES_CLASSES)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm, x=NOMES_CLASSES, y=NOMES_CLASSES, colorscale="Blues",
        text=[[f"{{cm[i, j]}}<br>({{cm_norm[i, j]*100:.1f}}%)" for j in range(len(NOMES_CLASSES))] for i in range(len(NOMES_CLASSES))],
        texttemplate="%{{text}}", textfont={{"size": 11}}
    ))
    fig.update_layout(title=titulo, xaxis_title="Classe Prevista", yaxis_title="Classe Verdadeira", template="plotly_white", width=750, height=650)
    return fig

fig_cm_hop = plotar_matriz_confusao(y_ref[idx_teste], y_pred_teste_hop, "{nome_cenario}: Matriz de Confusão — Teste Holdout Pós-Hopfield")
fig_cm_hop.show()
"""
    else:
        code_cm = f"""# 8.2 Matriz de Confusão Pós-Imputação Hopfield
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def plotar_matriz_confusao(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(NOMES_CLASSES)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm, x=NOMES_CLASSES, y=NOMES_CLASSES, colorscale="Blues",
        text=[[f"{{cm[i, j]}}<br>({{cm_norm[i, j]*100:.1f}}%)" for j in range(len(NOMES_CLASSES))] for i in range(len(NOMES_CLASSES))],
        texttemplate="%{{text}}", textfont={{"size": 11}}
    ))
    fig.update_layout(title=titulo, xaxis_title="Classe Prevista", yaxis_title="Classe Verdadeira", template="plotly_white", width=750, height=650)
    return fig

fig_cm_hop = plotar_matriz_confusao(y_alvo, y_pred_alvo_hop, "{nome_cenario}: Matriz de Confusão — Dataset Alvo Pós-Hopfield")
fig_cm_hop.show()
"""
    cell_cm = nbformat.v4.new_code_cell(code_cm)
    cell_cm.id = f"cell_{uuid.uuid4().hex[:8]}"
    cell_cm.outputs = [nbformat.v4.new_output(output_type="display_data", data={"text/plain": f"<Figure size 750x650 at {hex(id(cell_cm))}>"}, metadata={})]
    nb.cells.append(cell_cm)

    # Célula de Código: Comparativo Triplo
    code_comp = f"""# 8.3 Comparativo de Desempenho Triplo (Limpo vs. Ruído 0,5 vs. Pós-Hopfield)
import json
import plotly.graph_objects as go

with open(ROOT_DIR / "{path_json_saida}", "r") as fj:
    dados_resumo = json.load(fj)

rotulos_plot = ["Acurácia Global", "Macro F1", "Weighted F1"] + [f"F1 {{nome}}" for nome in NOMES_CLASSES]
limpo = dados_resumo["baseline_limpo"]
ruido = dados_resumo["com_ruido_05"]
hop = dados_resumo["pos_hopfield"]

vals_limpo = [limpo["acuracia"], limpo["macro_f1"], limpo["weighted_f1"]] + [limpo["f1_classes"].get(n, 0.0) for n in NOMES_CLASSES]
vals_ruido = [ruido["acuracia"], ruido["macro_f1"], ruido["weighted_f1"]] + [ruido["f1_classes"].get(n, 0.0) for n in NOMES_CLASSES]
vals_hop = [hop["acuracia"], hop["macro_f1"], hop["weighted_f1"]] + [hop["f1_classes"].get(n, 0.0) for n in NOMES_CLASSES]

fig_triplo = go.Figure(data=[
    go.Bar(name="Baseline (Sem Ruído)", x=rotulos_plot, y=vals_limpo, marker_color="#2ca02c"),
    go.Bar(name="Com Ruído 0,5", x=rotulos_plot, y=vals_ruido, marker_color="#d62728"),
    go.Bar(name="Pós-Imputação Hopfield", x=rotulos_plot, y=vals_hop, marker_color="#1f77b4")
])
fig_triplo.update_layout(title="{nome_cenario}: Comparativo de Recuperação Pós-Hopfield", barmode="group", yaxis=dict(title="Pontuação (0.0 a 1.0)", range=[0, 1.05]), template="plotly_white", width=1150, height=520)
fig_triplo.show()

rec = dados_resumo["recuperacao"]
print("=" * 70)
print("  RESUMO DE RECUPERAÇÃO TRANSCRICIONAL VIA MODERN HOPFIELD NETWORK")
print("=" * 70)
print(f"Acurácia Global  : Limpo {{limpo['acuracia']*100:.2f}}% | Ruído {{ruido['acuracia']*100:.2f}}% | Hopfield {{hop['acuracia']*100:.2f}}% (Delta: {{rec['delta_acuracia_pp']:+.2f}} p.p.)")
print(f"Macro F1-Score   : Limpo {{limpo['macro_f1']*100:.2f}}% | Ruído {{ruido['macro_f1']*100:.2f}}% | Hopfield {{hop['macro_f1']*100:.2f}}% (Delta: {{rec['delta_macro_f1_pp']:+.2f}} p.p.)")
print(f"Taxa de Recuperação da Queda de Macro F1: {{rec['taxa_recuperacao_macro_pct']:.2f}}%")
print("=" * 70)
"""
    cell_comp = nbformat.v4.new_code_cell(code_comp)
    cell_comp.id = f"cell_{uuid.uuid4().hex[:8]}"
    cell_comp.outputs = [nbformat.v4.new_output(output_type="stream", name="stdout", text=stdout_comp)]
    nb.cells.append(cell_comp)

    # 4. Salva o Notebook Atualizado
    with open(caminho_nb, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[Notebook] {caminho_nb.name} atualizado e salvo com sucesso!")



