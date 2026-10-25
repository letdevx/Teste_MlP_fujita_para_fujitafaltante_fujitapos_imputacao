"""Script de Execução Autônoma da Frente 3: Referência Pan.

Executa:
1. Otimização Bayesiana KerasTuner (15 trials) na GPU em Pan idx_treino/idx_val.
2. Extração dos protótipos ótimos e persistência de hopfield_pan.pt e metadados.
3. Calibração do limiar ótimo (threshold*) via curva Precision-Recall em idx_val.
4. Imputação e projeção SWeeP 600D em streaming na GPU para:
   - Pan (Intra): X_pan_pos_hopfield_600d.npy
   - Fujita (Cross): X_fujita_alinhado_pan_pos_hopfield_600d.npy
   - Mathys (Cross): X_mathys_alinhado_pan_pos_hopfield_600d.npy
5. Criação do notebook experimentos_hopfield/hopfield_03_pan.ipynb com histórico.
6. Anexação da Seção 8 nos notebooks exp04, exp05 e exp06.
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
    print("  INICIANDO FRENTE 3: REFERÊNCIA PAN (GPU CUDA)")
    print("=" * 70)
    print(f"Diretório raiz : {ROOT_DIR}")
    print(f"PyTorch CUDA   : {torch.cuda.get_device_name(0)}")
    print(f"SWeeP GPU      : {sweep.get_gpu_device_name()}")

    dados_fujita = ROOT_DIR / "dados" / "fujita"
    dados_mathys = ROOT_DIR / "dados" / "mathys"
    dados_pan = ROOT_DIR / "dados" / "pan"
    saida_hopfield = dados_pan / "hopfield"
    saida_hopfield.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento e Particionamento Pan
    print("\n[Passo 1/7] Carregando matriz scRNA-seq Pan (CSR Binarizada)...")
    path_h5ad_pan = dados_pan / "pan_anotado.h5ad"
    path_labels_pan = dados_pan / "PanNumerico.csv"
    X_pan_csr, y_pan = carregar_matriz_binaria_csr(path_h5ad_pan, path_labels_pan)

    idx_treino, idx_val, idx_teste = obter_particoes_canonicas(y_pan, seed=42)
    print(
        f"Pan Carregado: {X_pan_csr.shape} | "
        f"Treino: {len(idx_treino)} ({len(idx_treino)/len(y_pan)*100:.1f}%), "
        f"Val: {len(idx_val)} ({len(idx_val)/len(y_pan)*100:.1f}%), "
        f"Teste: {len(idx_teste)} ({len(idx_teste)/len(y_pan)*100:.1f}%)"
    )

    wswp_pan = np.load(dados_pan / "X_pan_600d.npy")[idx_treino].astype(np.float32)

    # 2. Otimização Bayesiana KerasTuner (15 trials)
    print("\n[Passo 2/7] Executando KerasTuner (15 trials) na GPU...")
    dir_kt = ROOT_DIR / "outputs_kt_pan"
    tuner = HopfieldKerasTuner(
        w0=X_pan_csr[idx_treino],
        wswp=wswp_pan,
        labels=y_pan[idx_treino] + 1,
        x_val=X_pan_csr[idx_val],
        y_val=y_pan[idx_val] + 1,
        classes=CLASSES_VALIDAS,
        algorithm="bayesian",
        objective="val_f1_macro",
        max_trials=15,
        directory=str(dir_kt),
        project_name="hopfield_pan",
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
    print("\n[Passo 3/7] Extraindo protótipos de memória e configurando ModernHopfieldNetwork...")
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

    path_checkpoint = saida_hopfield / "hopfield_pan.pt"
    rede.save(path_checkpoint)

    # 4. Calibração do Limiar Ótimo (threshold*) via Curva PR em idx_val
    print("\n[Passo 4/7] Calibrando limiar ótimo de binarização (threshold*) via curva PR em idx_val...")
    genes_fujita = carregar_ensembl_ids(dados_fujita / "featuresF.tsv")
    genes_mathys = carregar_ensembl_ids(dados_mathys / "featuresM.tsv")
    genes_pan = carregar_ensembl_ids(dados_pan / "featuresPANcorrigido.tsv")
    s_triple = set(genes_fujita) & set(genes_mathys) & set(genes_pan)

    mask_pan_ocultos = np.array([g not in s_triple for g in genes_pan], dtype=bool)
    print(f"Genes Pan: {len(genes_pan):,} | s_triple: {len(s_triple):,} | Ocultos: {mask_pan_ocultos.sum():,}")

    res_pr = calibrar_limiar_otimo_pr(
        rede=rede,
        X_val_csr=X_pan_csr[idx_val],
        mask_genes_ocultos=mask_pan_ocultos,
        batch_size=2048,
    )
    threshold_prob_otimo = res_pr["threshold_prob_otimo"]

    metadata_hopfield = {
        "dataset_referencia": "pan",
        "n_celulas_treino": len(idx_treino),
        "n_genes_ref": len(genes_pan),
        "melhores_hiperparametros": best_hp.values,
        "score_val_f1_macro": float(melhor_trial.score),
        "calibracao_pr": res_pr,
    }
    with open(saida_hopfield / "hopfield_pan_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_hopfield, f, indent=2)

    # 5. Imputação e Projeção SWeeP Streaming na GPU
    print("\n[Passo 5/7] Imputação e Projeção SWeeP 600D na GPU...")
    orthbase_pan = np.load(dados_pan / "orthbase_pan_600d.npy").astype(np.float32)

    # 5.1 Intra-Dataset: Pan
    print("\n--- Imputando e Projetando Pan (Intra) ---")
    X_pan_hop_600d, stats_pan = imputar_e_projetar_sweep_streaming(
        rede=rede,
        X_alvo_csr=X_pan_csr,
        genes_alvo=genes_pan,
        genes_ref=genes_pan,
        orthbase_ref=orthbase_pan,
        threshold_prob_otimo=threshold_prob_otimo,
        mask_forcar_sentinela_ref=mask_pan_ocultos,
        chunk_size=2048,
    )
    np.save(dados_pan / "X_pan_pos_hopfield_600d.npy", X_pan_hop_600d)

    del X_pan_csr
    gc.collect()

    # 5.2 Cross-Dataset: Fujita
    print("\n--- Imputando e Projetando Fujita (Cross) ---")
    path_h5ad_fujita = dados_fujita / "matrizFiltradaeNormalizadaF (1).h5ad"
    path_labels_fujita = dados_fujita / "cell_types_binarioF.txt"
    X_fujita_csr, _ = carregar_matriz_binaria_csr(path_h5ad_fujita, path_labels_fujita)

    X_fujita_hop_600d, stats_fujita = imputar_e_projetar_sweep_streaming(
        rede=rede,
        X_alvo_csr=X_fujita_csr,
        genes_alvo=genes_fujita,
        genes_ref=genes_pan,
        orthbase_ref=orthbase_pan,
        threshold_prob_otimo=threshold_prob_otimo,
        chunk_size=2048,
    )
    np.save(dados_fujita / "X_fujita_alinhado_pan_pos_hopfield_600d.npy", X_fujita_hop_600d)

    del X_fujita_csr
    gc.collect()

    # 5.3 Cross-Dataset: Mathys
    print("\n--- Imputando e Projetando Mathys (Cross) ---")
    path_h5ad_mathys = dados_mathys / "matriz_anotada_finalM.h5ad"
    path_labels_mathys = dados_mathys / "tipos_celulares_numericoMs.txt"
    X_mathys_csr, _ = carregar_matriz_binaria_csr(path_h5ad_mathys, path_labels_mathys)

    X_mathys_hop_600d, stats_mathys = imputar_e_projetar_sweep_streaming(
        rede=rede,
        X_alvo_csr=X_mathys_csr,
        genes_alvo=genes_mathys,
        genes_ref=genes_pan,
        orthbase_ref=orthbase_pan,
        threshold_prob_otimo=threshold_prob_otimo,
        chunk_size=2048,
    )
    np.save(dados_mathys / "X_mathys_alinhado_pan_pos_hopfield_600d.npy", X_mathys_hop_600d)

    del X_mathys_csr
    gc.collect()

    # 6. Criação do Notebook hopfield_03_pan.ipynb
    print("\n[Passo 6/7] Criando e salvando experimentos_hopfield/hopfield_03_pan.ipynb...")
    pasta_exp_hop = ROOT_DIR / "experimentos_hopfield"
    pasta_exp_hop.mkdir(parents=True, exist_ok=True)

    nb_hop = nbformat.v4.new_notebook()
    nb_hop.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"},
    }

    nb_hop.cells.append(nbformat.v4.new_markdown_cell(f"""# Treinamento e Calibração da Modern Hopfield Network — Referência Pan

Este notebook registra o treinamento, a otimização bayesiana (KerasTuner), a calibração do limiar ótimo ($threshold^*$) via curva Precision-Recall e a geração das projeções SWeeP 600D pós-imputação para a referência Pan.

- **Dataset de Referência:** Pan ({len(genes_pan):,} genes / 57.457 células)
- **Melhor Trial KerasTuner:** {melhor_trial.trial_id} (Val Macro F1 = {melhor_trial.score:.4f})
- **Hiperparâmetros:** `{best_hp.values}`
- **Limiar Ótimo de Binarização:** `threshold_prob* = {threshold_prob_otimo:.4f}` (F1 Gene = {res_pr['melhor_f1_gene']:.4f})
- **Projeções Geradas:**
  1. `dados/pan/X_pan_pos_hopfield_600d.npy` (Intra)
  2. `dados/fujita/X_fujita_alinhado_pan_pos_hopfield_600d.npy` (Cross)
  3. `dados/mathys/X_mathys_alinhado_pan_pos_hopfield_600d.npy` (Cross)
"""))

    nb_hop.cells.append(nbformat.v4.new_code_cell(f"""# Resumo de Execução e Métricas
import json
with open("../dados/pan/hopfield/hopfield_pan_metadata.json", "r") as f:
    meta = json.load(f)
print("Metadata carregado com sucesso:")
print(json.dumps(meta, indent=2))
"""))

    caminho_nb_hop = pasta_exp_hop / "hopfield_03_pan.ipynb"
    with open(caminho_nb_hop, "w", encoding="utf-8") as f:
        nbformat.write(nb_hop, f)
    print(f"Notebook salvo: {caminho_nb_hop}")

    # 7. Atualização dos Notebooks exp04, exp05 e exp06
    print("\n[Atualizando exp04_pan_para_pan.ipynb...]")
    anexar_secao_pos_hopfield_notebook(
        caminho_notebook=ROOT_DIR / "experimentos" / "exp04_pan_para_pan.ipynb",
        path_proj_pos_hopfield="dados/pan/X_pan_pos_hopfield_600d.npy",
        nome_cenario="Experimento 04: Pan para Pan (Intra-Dataset)",
        is_cross_dataset=False,
        path_json_saida="dados/pan/relatorio_recuperacao_exp04.json",
        path_modelo="dados/pan/modelo_mlp_exp04_pan_para_pan.keras",
        path_scaler="dados/pan/scaler_exp04.pkl",
        path_y="dados/pan/y_pan.npy",
        path_proj_limpo="dados/pan/X_pan_600d.npy",
        path_proj_ruido="dados/pan/X_pan_ruido_600d.npy",
    )

    print("\n[Atualizando exp05_pan_para_fujita.ipynb...]")
    anexar_secao_pos_hopfield_notebook(
        caminho_notebook=ROOT_DIR / "experimentos" / "exp05_pan_para_fujita.ipynb",
        path_proj_pos_hopfield="dados/fujita/X_fujita_alinhado_pan_pos_hopfield_600d.npy",
        nome_cenario="Experimento 05: Pan para Fujita (Cross-Dataset)",
        is_cross_dataset=True,
        path_json_saida="dados/fujita/relatorio_recuperacao_exp05.json",
        path_modelo="dados/pan/modelo_mlp_exp05_pan_para_fujita.keras",
        path_scaler="dados/pan/scaler_exp05.pkl",
        path_y="dados/fujita/y_fujita.npy",
        path_proj_limpo="dados/fujita/X_fujita_600d.npy",
        path_proj_ruido="dados/fujita/X_fujita_alinhado_pan_ruido_600d.npy",
    )

    print("\n[Atualizando exp06_pan_para_mathys.ipynb...]")
    anexar_secao_pos_hopfield_notebook(
        caminho_notebook=ROOT_DIR / "experimentos" / "exp06_pan_para_mathys.ipynb",
        path_proj_pos_hopfield="dados/mathys/X_mathys_alinhado_pan_pos_hopfield_600d.npy",
        nome_cenario="Experimento 06: Pan para Mathys (Cross-Dataset)",
        is_cross_dataset=True,
        path_json_saida="dados/mathys/relatorio_recuperacao_exp06.json",
        path_modelo="dados/pan/modelo_mlp_exp06_pan_para_mathys.keras",
        path_scaler="dados/pan/scaler_exp06.pkl",
        path_y="dados/mathys/y_mathys.npy",
        path_proj_limpo="dados/mathys/X_mathys_600d.npy",
        path_proj_ruido="dados/mathys/X_mathys_alinhado_pan_ruido_600d.npy",
    )

    tempo_total = time.time() - t_global
    print("\n" + "=" * 70)
    print(f"  FRENTE 3 (PAN) CONCLUÍDA COM SUCESSO EM {tempo_total:.2f}s ({tempo_total/60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
