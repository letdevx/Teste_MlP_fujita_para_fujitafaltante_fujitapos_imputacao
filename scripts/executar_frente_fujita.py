"""Script de Execução Autônoma da Frente 1: Referência Fujita.

Executa:
1. Otimização Bayesiana KerasTuner (15 trials) na GPU em Fujita idx_treino/idx_val.
2. Extração dos protótipos ótimos e persistência de hopfield_fujita.pt e metadados.
3. Calibração do limiar ótimo (threshold*) via curva Precision-Recall em idx_val.
4. Imputação e projeção SWeeP 600D em streaming na GPU para:
   - Fujita (Intra): X_fujita_pos_hopfield_600d.npy
   - Mathys (Cross): X_mathys_alinhado_fujita_pos_hopfield_600d.npy
5. Criação do notebook experimentos_hopfield/hopfield_01_fujita.ipynb com histórico.
6. Anexação da Seção 8 nos notebooks exp01_fujita_para_fujita.ipynb e exp02_fujita_para_mathys.ipynb.
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
    print("  INICIANDO FRENTE 1: REFERÊNCIA FUJITA (GPU CUDA)")
    print("=" * 70)
    print(f"Diretório raiz : {ROOT_DIR}")
    print(f"PyTorch CUDA   : {torch.cuda.get_device_name(0)}")
    print(f"SWeeP GPU      : {sweep.get_gpu_device_name()}")

    dados_fujita = ROOT_DIR / "dados" / "fujita"
    dados_mathys = ROOT_DIR / "dados" / "mathys"
    dados_pan = ROOT_DIR / "dados" / "pan"
    saida_hopfield = dados_fujita / "hopfield"
    saida_hopfield.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento e Particionamento Fujita
    print("\n[Passo 1/6] Carregando matriz scRNA-seq Fujita (CSR Binarizada)...")
    path_h5ad_fujita = dados_fujita / "matrizFiltradaeNormalizadaF (1).h5ad"
    path_labels_fujita = dados_fujita / "cell_types_binarioF.txt"
    X_fujita_csr, y_fujita = carregar_matriz_binaria_csr(path_h5ad_fujita, path_labels_fujita)

    idx_treino, idx_val, idx_teste = obter_particoes_canonicas(y_fujita, seed=42)
    print(
        f"Fujita Carregado: {X_fujita_csr.shape} | "
        f"Treino: {len(idx_treino)} ({len(idx_treino)/len(y_fujita)*100:.1f}%), "
        f"Val: {len(idx_val)} ({len(idx_val)/len(y_fujita)*100:.1f}%), "
        f"Teste: {len(idx_teste)} ({len(idx_teste)/len(y_fujita)*100:.1f}%)"
    )

    wswp_fujita = np.load(dados_fujita / "X_fujita_600d.npy")[idx_treino].astype(np.float32)

    # 2. Otimização Bayesiana KerasTuner (15 trials)
    print("\n[Passo 2/6] Executando KerasTuner (15 trials) na GPU...")
    dir_kt = ROOT_DIR / "outputs_kt_fujita"
    tuner = HopfieldKerasTuner(
        w0=X_fujita_csr[idx_treino],
        wswp=wswp_fujita,
        labels=y_fujita[idx_treino] + 1,
        x_val=X_fujita_csr[idx_val],
        y_val=y_fujita[idx_val] + 1,
        classes=CLASSES_VALIDAS,
        algorithm="bayesian",
        objective="val_f1_macro",
        max_trials=15,
        directory=str(dir_kt),
        project_name="hopfield_fujita",
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
    print("\n[Passo 3/6] Extraindo protótipos de memória e configurando ModernHopfieldNetwork...")
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

    path_checkpoint = saida_hopfield / "hopfield_fujita.pt"
    rede.save(path_checkpoint)

    # 4. Calibração do Limiar Ótimo (threshold*) via Curva PR em idx_val
    print("\n[Passo 4/6] Calibrando limiar ótimo de binarização (threshold*) via curva PR em idx_val...")
    genes_fujita = carregar_ensembl_ids(dados_fujita / "featuresF.tsv")
    genes_mathys = carregar_ensembl_ids(dados_mathys / "featuresM.tsv")
    genes_pan = carregar_ensembl_ids(dados_pan / "featuresPANcorrigido.tsv")
    s_triple = set(genes_fujita) & set(genes_mathys) & set(genes_pan)

    mask_fujita_ocultos = np.array([g not in s_triple for g in genes_fujita], dtype=bool)
    print(f"Genes Fujita: {len(genes_fujita):,} | s_triple: {len(s_triple):,} | Ocultos: {mask_fujita_ocultos.sum():,}")

    res_pr = calibrar_limiar_otimo_pr(
        rede=rede,
        X_val_csr=X_fujita_csr[idx_val],
        mask_genes_ocultos=mask_fujita_ocultos,
        batch_size=2048,
    )
    threshold_prob_otimo = res_pr["threshold_prob_otimo"]

    metadata_hopfield = {
        "dataset_referencia": "fujita",
        "n_celulas_treino": len(idx_treino),
        "n_genes_ref": len(genes_fujita),
        "melhores_hiperparametros": best_hp.values,
        "score_val_f1_macro": float(melhor_trial.score),
        "calibracao_pr": res_pr,
    }
    with open(saida_hopfield / "hopfield_fujita_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_hopfield, f, indent=2)

    # 5. Imputação e Projeção SWeeP Streaming na GPU
    print("\n[Passo 5/6] Imputação e Projeção SWeeP 600D na GPU...")
    orthbase_fujita = np.load(dados_fujita / "orthbase_fujita_600d.npy").astype(np.float32)

    # 5.1 Intra-Dataset: Fujita
    print("\n--- Imputando e Projetando Fujita (Intra) ---")
    X_fujita_hop_600d, stats_fujita = imputar_e_projetar_sweep_streaming(
        rede=rede,
        X_alvo_csr=X_fujita_csr,
        genes_alvo=genes_fujita,
        genes_ref=genes_fujita,
        orthbase_ref=orthbase_fujita,
        threshold_prob_otimo=threshold_prob_otimo,
        mask_forcar_sentinela_ref=mask_fujita_ocultos,
        chunk_size=2048,
    )
    np.save(dados_fujita / "X_fujita_pos_hopfield_600d.npy", X_fujita_hop_600d)

    # Libera memória Fujita CSR antes de ler Mathys
    del X_fujita_csr
    gc.collect()

    # 5.2 Cross-Dataset: Mathys
    print("\n--- Imputando e Projetando Mathys (Cross) ---")
    path_h5ad_mathys = dados_mathys / "matriz_anotada_finalM.h5ad"
    path_labels_mathys = dados_mathys / "tipos_celulares_numericoMs.txt"
    X_mathys_csr, _ = carregar_matriz_binaria_csr(path_h5ad_mathys, path_labels_mathys)

    X_mathys_hop_600d, stats_mathys = imputar_e_projetar_sweep_streaming(
        rede=rede,
        X_alvo_csr=X_mathys_csr,
        genes_alvo=genes_mathys,
        genes_ref=genes_fujita,
        orthbase_ref=orthbase_fujita,
        threshold_prob_otimo=threshold_prob_otimo,
        chunk_size=2048,
    )
    np.save(dados_mathys / "X_mathys_alinhado_fujita_pos_hopfield_600d.npy", X_mathys_hop_600d)

    del X_mathys_csr
    gc.collect()

    # 6. Criação do Notebook hopfield_01_fujita.ipynb
    print("\n[Passo 6/6] Criando e salvando experimentos_hopfield/hopfield_01_fujita.ipynb...")
    pasta_exp_hop = ROOT_DIR / "experimentos_hopfield"
    pasta_exp_hop.mkdir(parents=True, exist_ok=True)

    nb_hop = nbformat.v4.new_notebook()
    nb_hop.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"},
    }

    nb_hop.cells.append(nbformat.v4.new_markdown_cell(f"""# Treinamento e Calibração da Modern Hopfield Network — Referência Fujita

Este notebook registra o treinamento, a otimização bayesiana (KerasTuner), a calibração do limiar ótimo ($threshold^*$) via curva Precision-Recall e a geração das projeções SWeeP 600D pós-imputação para a referência Fujita.

- **Dataset de Referência:** Fujita ({len(genes_fujita):,} genes / 26.763 células)
- **Melhor Trial KerasTuner:** {melhor_trial.trial_id} (Val Macro F1 = {melhor_trial.score:.4f})
- **Hiperparâmetros:** `{best_hp.values}`
- **Limiar Ótimo de Binarização:** `threshold_prob* = {threshold_prob_otimo:.4f}` (F1 Gene = {res_pr['melhor_f1_gene']:.4f})
- **Projeções Geradas:**
  1. `dados/fujita/X_fujita_pos_hopfield_600d.npy` (Intra)
  2. `dados/mathys/X_mathys_alinhado_fujita_pos_hopfield_600d.npy` (Cross)
"""))

    nb_hop.cells.append(nbformat.v4.new_code_cell(f"""# Resumo de Execução e Métricas
import json
with open("../dados/fujita/hopfield/hopfield_fujita_metadata.json", "r") as f:
    meta = json.load(f)
print("Metadata carregado com sucesso:")
print(json.dumps(meta, indent=2))
"""))

    caminho_nb_hop = pasta_exp_hop / "hopfield_01_fujita.ipynb"
    with open(caminho_nb_hop, "w", encoding="utf-8") as f:
        nbformat.write(nb_hop, f)
    print(f"Notebook salvo: {caminho_nb_hop}")

    # 7. Atualização dos Notebooks exp01 e exp02
    print("\n[Atualizando exp01_fujita_para_fujita.ipynb...]")
    anexar_secao_pos_hopfield_notebook(
        caminho_notebook=ROOT_DIR / "experimentos" / "exp01_fujita_para_fujita.ipynb",
        path_proj_pos_hopfield="dados/fujita/X_fujita_pos_hopfield_600d.npy",
        nome_cenario="Experimento 01: Fujita para Fujita (Intra-Dataset)",
        is_cross_dataset=False,
        path_json_saida="dados/fujita/relatorio_recuperacao_exp01.json",
        path_modelo="dados/fujita/modelo_mlp_exp01_fujita_para_fujita.keras",
        path_scaler="dados/fujita/scaler_exp01.pkl",
        path_y="dados/fujita/y_fujita.npy",
        path_proj_limpo="dados/fujita/X_fujita_600d.npy",
        path_proj_ruido="dados/fujita/X_fujita_ruido_600d.npy",
    )

    print("\n[Atualizando exp02_fujita_para_mathys.ipynb...]")
    anexar_secao_pos_hopfield_notebook(
        caminho_notebook=ROOT_DIR / "experimentos" / "exp02_fujita_para_mathys.ipynb",
        path_proj_pos_hopfield="dados/mathys/X_mathys_alinhado_fujita_pos_hopfield_600d.npy",
        nome_cenario="Experimento 02: Fujita para Mathys (Cross-Dataset)",
        is_cross_dataset=True,
        path_json_saida="dados/mathys/relatorio_recuperacao_exp02.json",
        path_modelo="dados/fujita/modelo_mlp_exp02_fujita_para_mathys.keras",
        path_scaler="dados/fujita/scaler_exp02.pkl",
        path_y="dados/mathys/y_mathys.npy",
        path_proj_limpo="dados/mathys/X_mathys_600d.npy",
        path_proj_ruido="dados/mathys/X_mathys_alinhado_fujita_ruido_600d.npy",
    )

    tempo_total = time.time() - t_global
    print("\n" + "=" * 70)
    print(f"  FRENTE 1 (FUJITA) CONCLUÍDA COM SUCESSO EM {tempo_total:.2f}s ({tempo_total/60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
