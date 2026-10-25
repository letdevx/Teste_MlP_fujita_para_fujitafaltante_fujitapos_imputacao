"""Script de Execução Autônoma da Frente 2: Referência Mathys.

Executa:
1. Otimização Bayesiana KerasTuner (15 trials) na GPU em Mathys idx_treino/idx_val.
2. Extração dos protótipos ótimos e persistência de hopfield_mathys.pt e metadados.
3. Calibração do limiar ótimo (threshold*) via curva Precision-Recall em idx_val.
4. Imputação e projeção SWeeP 600D em streaming na GPU para:
   - Mathys (Intra): X_mathys_pos_hopfield_600d.npy
5. Criação do notebook experimentos_hopfield/hopfield_02_mathys.ipynb com histórico.
6. Anexação da Seção 8 no notebook exp03_mathys_para_mathys.ipynb.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Restrições de CPU e Backend
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KERAS_BACKEND"] = "torch"

import cupy as cp
import nbformat
import numpy as np
import scipy.sparse as sp
import sweep
import torch

torch.set_num_threads(1)

# Caminho raiz
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.hopfield.hopfield import ModernHopfieldNetwork
from src.hopfield.hopfield_kerastuner import HopfieldKerasTuner
from src.hopfield.pipeline_imputacao import (
    CLASSES_VALIDAS,
    anexar_secao_pos_hopfield_notebook,
    calibrar_limiar_otimo_pr,
    carregar_ensembl_ids,
    carregar_matriz_binaria_csr,
    imputar_e_projetar_sweep_streaming,
    obter_particoes_canonicas,
)


def main() -> None:
    t_global = time.time()
    print("=" * 70)
    print("  INICIANDO FRENTE 2: REFERÊNCIA MATHYS (GPU CUDA)")
    print("=" * 70)
    print(f"Diretório raiz : {ROOT_DIR}")
    print(f"PyTorch CUDA   : {torch.cuda.get_device_name(0)}")
    print(f"SWeeP GPU      : {sweep.get_gpu_device_name()}")

    dados_fujita = ROOT_DIR / "dados" / "fujita"
    dados_mathys = ROOT_DIR / "dados" / "mathys"
    dados_pan = ROOT_DIR / "dados" / "pan"
    saida_hopfield = dados_mathys / "hopfield"
    saida_hopfield.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento e Particionamento Mathys
    print("\n[Passo 1/5] Carregando matriz scRNA-seq Mathys (CSR Binarizada)...")
    path_h5ad_mathys = dados_mathys / "matriz_anotada_finalM.h5ad"
    path_labels_mathys = dados_mathys / "tipos_celulares_numericoMs.txt"
    X_mathys_csr, y_mathys = carregar_matriz_binaria_csr(path_h5ad_mathys, path_labels_mathys)

    idx_treino, idx_val, idx_teste = obter_particoes_canonicas(y_mathys, seed=42)
    print(
        f"Mathys Carregado: {X_mathys_csr.shape} | "
        f"Treino: {len(idx_treino)} ({len(idx_treino)/len(y_mathys)*100:.1f}%), "
        f"Val: {len(idx_val)} ({len(idx_val)/len(y_mathys)*100:.1f}%), "
        f"Teste: {len(idx_teste)} ({len(idx_teste)/len(y_mathys)*100:.1f}%)"
    )

    wswp_mathys = np.load(dados_mathys / "X_mathys_600d.npy")[idx_treino].astype(np.float32)

    # 2. Otimização Bayesiana KerasTuner (15 trials)
    print("\n[Passo 2/5] Executando KerasTuner (15 trials) na GPU...")
    dir_kt = ROOT_DIR / "outputs_kt_mathys"
    tuner = HopfieldKerasTuner(
        w0=X_mathys_csr[idx_treino],
        wswp=wswp_mathys,
        labels=y_mathys[idx_treino] + 1,
        x_val=X_mathys_csr[idx_val],
        y_val=y_mathys[idx_val] + 1,
        classes=CLASSES_VALIDAS,
        algorithm="bayesian",
        objective="val_f1_macro",
        max_trials=15,
        directory=str(dir_kt),
        project_name="hopfield_mathys",
        overwrite=True,
        seed=42,
    )
    tuner.search()

    melhor_trial = tuner.oracle.get_best_trials(1)[0]
    best_hp = melhor_trial.hyperparameters
    print(f"\nMelhor Configuração Encontrada (Trial {melhor_trial.trial_id}):")
    print(f"  val_f1_macro = {melhor_trial.score:.4f}")
    for k, v in best_hp.values.items():
        print(f"  {k}: {v}")

    # 3. Extração dos Melhores Protótipos e Armazenamento
    print("\n[Passo 3/5] Extraindo protótipos de memória e configurando ModernHopfieldNetwork...")
    padroes, meta = tuner.obter_padroes(
        nc=int(best_hp.get("nc")),
        k_vizinhos=int(best_hp.get("k_vizinhos")),
        estrategia=best_hp.get("estrategia"),
    )
    rede = ModernHopfieldNetwork(
        beta=float(best_hp.get("beta")),
        threshold=float(best_hp.get("threshold")),
        normalize=bool(best_hp.get("normalize")),
        scale_by_dim=bool(best_hp.get("scale_by_dim")),
        binary=True,
    )
    rede.store(padroes)

    path_checkpoint = saida_hopfield / "hopfield_mathys.pt"
    rede.save(path_checkpoint)

    # 4. Calibração do Limiar Ótimo (threshold*) via Curva PR em idx_val
    print("\n[Passo 4/5] Calibrando limiar ótimo de binarização (threshold*) via curva PR em idx_val...")
    genes_fujita = carregar_ensembl_ids(dados_fujita / "featuresF.tsv")
    genes_mathys = carregar_ensembl_ids(dados_mathys / "featuresM.tsv")
    genes_pan = carregar_ensembl_ids(dados_pan / "featuresPANcorrigido.tsv")
    s_triple = set(genes_fujita) & set(genes_mathys) & set(genes_pan)

    mask_mathys_ocultos = np.array([g not in s_triple for g in genes_mathys], dtype=bool)
    print(f"Genes Mathys: {len(genes_mathys):,} | s_triple: {len(s_triple):,} | Ocultos: {mask_mathys_ocultos.sum():,}")

    res_pr = calibrar_limiar_otimo_pr(
        rede=rede,
        X_val_csr=X_mathys_csr[idx_val],
        mask_genes_ocultos=mask_mathys_ocultos,
        batch_size=2048,
    )
    threshold_prob_otimo = res_pr["threshold_prob_otimo"]

    metadata_hopfield = {
        "dataset_referencia": "mathys",
        "n_celulas_treino": len(idx_treino),
        "n_genes_ref": len(genes_mathys),
        "melhores_hiperparametros": best_hp.values,
        "score_val_f1_macro": float(melhor_trial.score),
        "calibracao_pr": res_pr,
    }
    with open(saida_hopfield / "hopfield_mathys_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_hopfield, f, indent=2)

    # 5. Imputação e Projeção SWeeP Streaming na GPU (Mathys Intra)
    print("\n[Passo 5/5] Imputação e Projeção SWeeP 600D na GPU (Mathys Intra)...")
    orthbase_mathys = np.load(dados_mathys / "orthbase_mathys_600d.npy").astype(np.float32)

    X_mathys_hop_600d, stats_mathys = imputar_e_projetar_sweep_streaming(
        rede=rede,
        X_alvo_csr=X_mathys_csr,
        genes_alvo=genes_mathys,
        genes_ref=genes_mathys,
        orthbase_ref=orthbase_mathys,
        threshold_prob_otimo=threshold_prob_otimo,
        mask_forcar_sentinela_ref=mask_mathys_ocultos,
        chunk_size=2048,
    )
    np.save(dados_mathys / "X_mathys_pos_hopfield_600d.npy", X_mathys_hop_600d)

    del X_mathys_csr
    gc.collect()

    # 6. Criação do Notebook hopfield_02_mathys.ipynb
    print("\nCriando e salvando experimentos_hopfield/hopfield_02_mathys.ipynb...")
    pasta_exp_hop = ROOT_DIR / "experimentos_hopfield"
    pasta_exp_hop.mkdir(parents=True, exist_ok=True)

    nb_hop = nbformat.v4.new_notebook()
    nb_hop.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"},
    }

    nb_hop.cells.append(nbformat.v4.new_markdown_cell(f"""# Treinamento e Calibração da Modern Hopfield Network — Referência Mathys

Este notebook registra o treinamento, a otimização bayesiana (KerasTuner), a calibração do limiar ótimo ($threshold^*$) via curva Precision-Recall e a geração da projeção SWeeP 600D pós-imputação para a referência Mathys.

- **Dataset de Referência:** Mathys ({len(genes_mathys):,} genes / 45.572 células)
- **Melhor Trial KerasTuner:** {melhor_trial.trial_id} (Val Macro F1 = {melhor_trial.score:.4f})
- **Hiperparâmetros:** `{best_hp.values}`
- **Limiar Ótimo de Binarização:** `threshold_prob* = {threshold_prob_otimo:.4f}` (F1 Gene = {res_pr['melhor_f1_gene']:.4f})
- **Projeção Gerada:** `dados/mathys/X_mathys_pos_hopfield_600d.npy` (Intra)
"""))

    nb_hop.cells.append(nbformat.v4.new_code_cell(f"""# Resumo de Execução e Métricas
import json
with open("../dados/mathys/hopfield/hopfield_mathys_metadata.json", "r") as f:
    meta = json.load(f)
print("Metadata carregado com sucesso:")
print(json.dumps(meta, indent=2))
"""))

    caminho_nb_hop = pasta_exp_hop / "hopfield_02_mathys.ipynb"
    with open(caminho_nb_hop, "w", encoding="utf-8") as f:
        nbformat.write(nb_hop, f)
    print(f"Notebook salvo: {caminho_nb_hop}")

    # 7. Atualização do Notebook exp03
    print("\n[Atualizando exp03_mathys_para_mathys.ipynb...]")
    anexar_secao_pos_hopfield_notebook(
        caminho_notebook=ROOT_DIR / "experimentos" / "exp03_mathys_para_mathys.ipynb",
        path_proj_pos_hopfield="dados/mathys/X_mathys_pos_hopfield_600d.npy",
        nome_cenario="Experimento 03: Mathys para Mathys (Intra-Dataset)",
        is_cross_dataset=False,
        path_json_saida="dados/mathys/relatorio_recuperacao_exp03.json",
        path_modelo="dados/mathys/modelo_mlp_exp03_mathys_para_mathys.keras",
        path_scaler="dados/mathys/scaler_exp03.pkl",
        path_y="dados/mathys/y_mathys.npy",
        path_proj_limpo="dados/mathys/X_mathys_600d.npy",
        path_proj_ruido="dados/mathys/X_mathys_ruido_600d.npy",
    )

    tempo_total = time.time() - t_global
    print("\n" + "=" * 70)
    print(f"  FRENTE 2 (MATHYS) CONCLUÍDA COM SUCESSO EM {tempo_total:.2f}s ({tempo_total/60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
