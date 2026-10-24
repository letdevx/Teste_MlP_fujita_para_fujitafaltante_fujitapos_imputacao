import json
from pathlib import Path

ROOT = Path("/home/rasert/workspace/Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao")
EXP_DIR = ROOT / "experimentos"
EXP_DIR.mkdir(exist_ok=True)

def create_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

import uuid

def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "cell_" + str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell_" + str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# Template comum de cabeçalho e imports
HEADER_CODE = '''import gc
import json
import os
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sweep
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, load_model

# Garantir que o diretório de execução seja a raiz do projeto
if Path.cwd().name == "experimentos":
    os.chdir(Path.cwd().parent)
ROOT_DIR = Path.cwd()
DADOS_DIR = ROOT_DIR / "dados"

SEED = 42
DIM_PROJECAO = 600
CLASSES_VALIDAS = [1, 2, 3, 4, 5, 6, 7]
NOMES_CLASSES = [f"Classe {i}" for i in CLASSES_VALIDAS]

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

assert sweep.is_gpu_available(), "ERRO: GPU CUDA não detectada pelo pacote sweep!"
print(f"Diretório raiz ativo: {ROOT_DIR}")
print(f"Dispositivo GPU     : {sweep.get_gpu_device_name()}")
print(f"Versão SWeeP        : {getattr(sweep, 'DEFAULT_VERSION', 'v2.2.0')}")
'''

UTILS_CODE = '''def carregar_ensembl_ids(path_tsv: Path) -> list[str]:
    df = pd.read_csv(path_tsv, sep="\\t", header=None)
    return df[0].astype(str).str.strip().str.replace(r"\\.\\d+$", "", regex=True).tolist()

def carregar_rotulos_filtrados(path_labels: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path_labels, "r", encoding="utf-8") as f:
        rotulos = np.array([int(float(l.strip().split(",")[0])) for l in f if l.strip()], dtype=np.int64)
    mascara = np.isin(rotulos, CLASSES_VALIDAS)
    y_filtrado = rotulos[mascara] - 1  # 1..7 -> 0..6
    return mascara, y_filtrado

def construir_modelo_mlp(input_dim: int = 600, n_classes: int = 7) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu", name="camada_oculta_1"),
        Dense(64, activation="relu", name="camada_oculta_2"),
        Dense(32, activation="relu", name="camada_oculta_3"),
        Dense(n_classes, activation="softmax", name="camada_saida"),
    ], name="MLP_Classificador_Celular_600D")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plotar_curvas_aprendizado(history, titulo: str):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Acurácia ao Longo das Épocas", "Perda (Loss) ao Longo das Épocas")
    )
    epocas = list(range(1, len(history.history["accuracy"]) + 1))
    fig.add_trace(go.Scatter(x=epocas, y=history.history["accuracy"], mode="lines+markers", name="Treino Acurácia", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epocas, y=history.history["val_accuracy"], mode="lines+markers", name="Validação Acurácia", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epocas, y=history.history["loss"], mode="lines+markers", name="Treino Loss", line=dict(color="#1f77b4")), row=1, col=2)
    fig.add_trace(go.Scatter(x=epocas, y=history.history["val_loss"], mode="lines+markers", name="Validação Loss", line=dict(color="#ff7f0e")), row=1, col=2)
    fig.update_layout(title_text=titulo, template="plotly_white", width=1000, height=450, legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(title_text="Época", dtick=1)
    fig.update_yaxes(title_text="Acurácia", row=1, col=1)
    fig.update_yaxes(title_text="Perda", row=1, col=2)
    return fig

def plotar_matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray, titulo: str):
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_abs = confusion_matrix(y_true, y_pred)
    text_labels = [[f"{abs_v}<br>({norm_v*100:.1f}%)" for abs_v, norm_v in zip(r_abs, r_norm)] for r_abs, r_norm in zip(cm_abs, cm_norm)]
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=NOMES_CLASSES,
        y=NOMES_CLASSES,
        colorscale="Blues",
        text=text_labels,
        texttemplate="%{text}",
        colorbar=dict(title="Recall"),
        zmin=0, zmax=1
    ))
    fig.update_layout(
        title=titulo,
        xaxis_title="Classe Predita",
        yaxis_title="Classe Real",
        yaxis_autorange="reversed",
        template="plotly_white",
        width=650, height=550
    )
    return fig

def plotar_comparativo(rel_limpo: dict, rel_ruido: dict, titulo: str, nome_cenario_ruido: str = "Com Ruído 0,5"):
    rotulos_plot = ["Acurácia Global", "Macro F1", "Weighted F1"] + [f"F1 {nome}" for nome in NOMES_CLASSES]
    
    vals_limpo = [
        rel_limpo["accuracy"],
        rel_limpo["macro avg"]["f1-score"],
        rel_limpo["weighted avg"]["f1-score"]
    ] + [rel_limpo.get(nome, {}).get("f1-score", 0.0) for nome in NOMES_CLASSES]
    
    vals_ruido = [
        rel_ruido["accuracy"],
        rel_ruido["macro avg"]["f1-score"],
        rel_ruido["weighted avg"]["f1-score"]
    ] + [rel_ruido.get(nome, {}).get("f1-score", 0.0) for nome in NOMES_CLASSES]
    
    fig = go.Figure(data=[
        go.Bar(name="Baseline (Sem Ruído)", x=rotulos_plot, y=vals_limpo, marker_color="#2ca02c"),
        go.Bar(name=nome_cenario_ruido, x=rotulos_plot, y=vals_ruido, marker_color="#d62728")
    ])
    fig.update_layout(
        title=titulo,
        barmode="group",
        yaxis=dict(title="Pontuação (0.0 a 1.0)", range=[0, 1.05]),
        template="plotly_white",
        width=1100, height=500,
        legend=dict(orientation="h", y=1.12)
    )
    return fig
'''

print("Configurações base prontas.")

# ==============================================================================
# GERADOR DO EXPERIMENTO 01: Fujita -> Fujita (Intra)
# ==============================================================================
def gerar_exp01():
    cells = [
        md_cell("""# Experimento 01: Validação de Sensibilidade ao Ruído (0,5) — Fujita para Fujita (Intra-Dataset)

## Objetivo
Avaliar se a inserção do ruído sentinela neutro (`0,5`) nos genes do **Fujita** que estão ausentes em pelo menos um dos outros dois datasets (**Pan** ou **Mathys**) degrada a capacidade preditiva do classificador celular **Multi-Layer Perceptron (MLP)** treinado sobre as projeções limpas do SWeeP 600D.

- **Dataset de Treino (Referência Limpa):** Fujita (`26.763` células válidas, classes 1–7)
- **Dataset de Teste com Ruído:** Fujita com `0,5` nas colunas ausentes na união de ausências (`G_Fujita \\ (G_Pan ∩ G_Fujita ∩ G_Mathys)`)
- **Espaço de Projeção:** `36.601` genes $\\to$ 600D com `orthbase_fujita_600d.npy`
- **Validação:** Teste Holdout (15% sem vazamento) e Dataset Completo (100%)
"""),
        code_cell(HEADER_CODE),
        code_cell(UTILS_CODE),
        code_cell("""# 1. Carregamento dos Dados, Rótulos e Ortobase
PATH_GENES_FUJITA = DADOS_DIR / "fujita" / "featuresF.tsv"
PATH_GENES_PAN = DADOS_DIR / "pan" / "featuresPANcorrigido.tsv"
PATH_GENES_MATHYS = DADOS_DIR / "mathys" / "featuresM.tsv"

genes_fujita = carregar_ensembl_ids(PATH_GENES_FUJITA)
genes_pan = carregar_ensembl_ids(PATH_GENES_PAN)
genes_mathys = carregar_ensembl_ids(PATH_GENES_MATHYS)

s_triple = set(genes_pan) & set(genes_fujita) & set(genes_mathys)
mask_ruido = np.array([g not in s_triple for g in genes_fujita])
print(f"Total genes Fujita : {len(genes_fujita)}")
print(f"Genes na interseção: {len(s_triple)}")
print(f"Genes com ruído 0.5: {mask_ruido.sum()} ({mask_ruido.sum()/len(genes_fujita)*100:.2f}%)")

orthbase_ref = np.load(DADOS_DIR / "fujita" / "orthbase_fujita_600d.npy").astype(np.float32)
X_limpo = np.load(DADOS_DIR / "fujita" / "X_fujita_600d.npy").astype(np.float32)
y_ref = np.load(DADOS_DIR / "fujita" / "y_fujita.npy")
X_ruido = np.load(DADOS_DIR / "fujita" / "X_fujita_ruido_600d.npy").astype(np.float32)

# TRAVAS DE SEGURANÇA (ASSERTIONS FAIL-FAST)
assert orthbase_ref.shape == (len(genes_fujita), DIM_PROJECAO), f"Ortobase inválida: {orthbase_ref.shape}"
assert X_limpo.shape == (len(y_ref), DIM_PROJECAO), f"X_limpo inválido: {X_limpo.shape}"
assert X_ruido.shape == (len(y_ref), DIM_PROJECAO), f"X_ruido inválido: {X_ruido.shape}"
print("Todas as travas de integridade passaram com sucesso!")
"""),
        code_cell("""# 2. Particionamento Estratificado (70% Treino / 15% Validação / 15% Teste Holdout)
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=y_ref
)
razao_val = 0.15 / (1.0 - 0.15)
idx_treino, idx_val = train_test_split(
    idx_treino_val, test_size=razao_val, random_state=SEED, stratify=y_ref[idx_treino_val]
)

print(f"Treino    : {len(idx_treino)} células ({len(idx_treino)/len(indices)*100:.1f}%)")
print(f"Validação : {len(idx_val)} células ({len(idx_val)/len(indices)*100:.1f}%)")
print(f"Teste     : {len(idx_teste)} células ({len(idx_teste)/len(indices)*100:.1f}%)")

# Escalonamento StandardScaler (Ajustado EXCLUSIVAMENTE no Treino Limpo)
scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_limpo[idx_treino])
X_val_s = scaler.transform(X_limpo[idx_val])
X_teste_limpo_s = scaler.transform(X_limpo[idx_teste])

# Aplica o MESMO scaler nos dados com ruído
X_teste_ruido_s = scaler.transform(X_ruido[idx_teste])
X_full_ruido_s = scaler.transform(X_ruido)

joblib.dump(scaler, DADOS_DIR / "fujita" / "scaler_exp01.pkl")
"""),
        code_cell("""# 3. Treinamento do Classificador MLP
modelo = construir_modelo_mlp(input_dim=DIM_PROJECAO, n_classes=len(CLASSES_VALIDAS))
modelo.summary()

caminho_modelo = DADOS_DIR / "fujita" / "modelo_mlp_exp01_fujita_para_fujita.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(caminho_modelo), monitor="val_loss", save_best_only=True, verbose=0)
]

t0 = time.time()
historico = modelo.fit(
    X_treino_s, y_ref[idx_treino],
    validation_data=(X_val_s, y_ref[idx_val]),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print(f"Treinamento concluído em {time.time() - t0:.2f} segundos.")
"""),
        code_cell("""# 4. Gráfico 1 — Curvas de Aprendizado (Acurácia e Loss)
fig_aprendizado = plotar_curvas_aprendizado(historico, "Experimento 01: Curvas de Aprendizado MLP (Fujita Limpo)")
fig_aprendizado.show()
"""),
        code_cell("""# 5. Avaliação Baseline Limpa (Conjunto de Teste Holdout 15%)
probs_teste_limpo = modelo.predict(X_teste_limpo_s, verbose=0)
y_pred_teste_limpo = np.argmax(probs_teste_limpo, axis=1)

relatorio_limpo = classification_report(
    y_ref[idx_teste], y_pred_teste_limpo,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO DE CLASSIFICAÇÃO: BASELINE LIMPO (TESTE 15%) ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_limpo, target_names=NOMES_CLASSES))

fig_cm_limpo = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_limpo,
    "Experimento 01: Matriz de Confusão — Teste Holdout Limpo (Baseline)"
)
fig_cm_limpo.show()
"""),
        code_cell("""# 6. Avaliação com Ruído Sentinela 0,5 (Teste Holdout 15% e Dataset Completo 100%)
# 6.1 Teste Holdout com Ruído
probs_teste_ruido = modelo.predict(X_teste_ruido_s, verbose=0)
y_pred_teste_ruido = np.argmax(probs_teste_ruido, axis=1)

relatorio_ruido = classification_report(
    y_ref[idx_teste], y_pred_teste_ruido,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO: TESTE HOLDOUT COM RUÍDO 0,5 ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_ruido, target_names=NOMES_CLASSES))

fig_cm_ruido = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_ruido,
    "Experimento 01: Matriz de Confusão — Teste Holdout com Ruído 0,5"
)
fig_cm_ruido.show()

# 6.2 Dataset Completo 100% com Ruído
probs_full_ruido = modelo.predict(X_full_ruido_s, verbose=0)
y_pred_full_ruido = np.argmax(probs_full_ruido, axis=1)
print(f"Acurácia no Dataset Completo (100%) com Ruído: {accuracy_score(y_ref, y_pred_full_ruido):.4f}")
"""),
        code_cell("""# 7. Gráfico 4 — Comparativo de Desempenho (Baseline Limpo vs. Ruído 0,5)
fig_comp = plotar_comparativo(
    relatorio_limpo, relatorio_ruido,
    "Experimento 01 (Fujita -> Fujita): Queda de Desempenho com Ruído 0,5",
    nome_cenario_ruido="Fujita com Ruído 0,5"
)
fig_comp.show()

acc_limpo = relatorio_limpo["accuracy"]
acc_ruido = relatorio_ruido["accuracy"]
macro_limpo = relatorio_limpo["macro avg"]["f1-score"]
macro_ruido = relatorio_ruido["macro avg"]["f1-score"]

print("=" * 60)
print(f"Acurácia Global   : {acc_limpo*100:.2f}% -> {acc_ruido*100:.2f}% (Queda de {(acc_limpo - acc_ruido)*100:+.2f} p.p.)")
print(f"Macro F1-Score    : {macro_limpo*100:.2f}% -> {macro_ruido*100:.2f}% (Queda de {(macro_limpo - macro_ruido)*100:+.2f} p.p.)")
print("=" * 60)
""")
    ]
    with open(EXP_DIR / "exp01_fujita_para_fujita.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, indent=1, ensure_ascii=False)
    print("exp01_fujita_para_fujita.ipynb criado.")


# ==============================================================================
# GERADOR DO EXPERIMENTO 02: Fujita -> Mathys (Cross)
# ==============================================================================
def gerar_exp02():
    cells = [
        md_cell("""# Experimento 02: Validação de Sensibilidade ao Ruído (0,5) — Fujita para Mathys (Cross-Dataset)

## Objetivo
Avaliar a capacidade de generalização e tolerância ao ruído do classificador celular **MLP treinado na referência limpa Fujita**, ao realizar inferência sobre o dataset **Mathys**, cujos genes não medidos no Mathys recebem o valor sentinela `0,5` no espaço de colunas do Fujita.

- **Dataset de Treino (Referência Limpa):** Fujita (`26.763` células válidas, classes 1–7)
- **Dataset de Inferência Alvo:** Mathys (`45.572` células válidas, classes 1–7) alinhado ao Fujita
- **Espaço de Projeção:** `36.601` genes $\\to$ 600D com **`orthbase_fujita_600d.npy`** (REUSO ESTRITO DA BASE DA REFERÊNCIA)
- **Ruído Injetado:** Genes do Fujita ausentes no Mathys recebem `0,5`
"""),
        code_cell(HEADER_CODE),
        code_cell(UTILS_CODE),
        code_cell("""# 1. Carregamento dos Dados, Rótulos e Ortobase da Referência
PATH_GENES_FUJITA = DADOS_DIR / "fujita" / "featuresF.tsv"
PATH_GENES_MATHYS = DADOS_DIR / "mathys" / "featuresM.tsv"

genes_fujita = carregar_ensembl_ids(PATH_GENES_FUJITA)
genes_mathys = carregar_ensembl_ids(PATH_GENES_MATHYS)

s_mathys = set(genes_mathys)
genes_comuns = [g for g in genes_fujita if g in s_mathys]
genes_ausentes = [g for g in genes_fujita if g not in s_mathys]
print(f"Referência (Fujita) : {len(genes_fujita)} genes")
print(f"Alvo (Mathys)       : {len(genes_mathys)} genes")
print(f"Genes em Comum      : {len(genes_comuns)}")
print(f"Genes Ausentes (0.5): {len(genes_ausentes)} ({len(genes_ausentes)/len(genes_fujita)*100:.2f}%)")

# Base Ortonormal OBRIGATORIAMENTE da REFERÊNCIA (Fujita)
orthbase_ref = np.load(DADOS_DIR / "fujita" / "orthbase_fujita_600d.npy").astype(np.float32)
X_ref_limpo = np.load(DADOS_DIR / "fujita" / "X_fujita_600d.npy").astype(np.float32)
y_ref = np.load(DADOS_DIR / "fujita" / "y_fujita.npy")

# Projeção do Alvo (Mathys) no Espaço da Referência
X_alvo_ruido = np.load(DADOS_DIR / "mathys" / "X_mathys_alinhado_fujita_ruido_600d.npy").astype(np.float32)
y_alvo = np.load(DADOS_DIR / "mathys" / "y_mathys.npy")

# TRAVAS DE SEGURANÇA (ASSERTIONS FAIL-FAST)
assert orthbase_ref.shape == (len(genes_fujita), DIM_PROJECAO), f"Ortobase inválida: {orthbase_ref.shape}"
assert X_ref_limpo.shape == (len(y_ref), DIM_PROJECAO), f"X_ref inválido: {X_ref_limpo.shape}"
assert X_alvo_ruido.shape == (len(y_alvo), DIM_PROJECAO), f"X_alvo inválido: {X_alvo_ruido.shape}"
print("Travas de segurança e auditoria de ortobase passaram com 100% de sucesso!")
"""),
        code_cell("""# 2. Particionamento e Escalonamento na Referência Limpa (Fujita)
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=y_ref
)
razao_val = 0.15 / (1.0 - 0.15)
idx_treino, idx_val = train_test_split(
    idx_treino_val, test_size=razao_val, random_state=SEED, stratify=y_ref[idx_treino_val]
)

scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_ref_limpo[idx_treino])
X_val_s = scaler.transform(X_ref_limpo[idx_val])
X_teste_limpo_s = scaler.transform(X_ref_limpo[idx_teste])

# Escalonamento dos dados do Alvo (Mathys) usando o Scaler da Referência
X_alvo_ruido_s = scaler.transform(X_alvo_ruido)

joblib.dump(scaler, DADOS_DIR / "fujita" / "scaler_exp02.pkl")
"""),
        code_cell("""# 3. Treinamento do Classificador MLP na Referência Limpa (Fujita)
modelo = construir_modelo_mlp(input_dim=DIM_PROJECAO, n_classes=len(CLASSES_VALIDAS))
caminho_modelo = DADOS_DIR / "fujita" / "modelo_mlp_exp02_fujita_para_mathys.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(caminho_modelo), monitor="val_loss", save_best_only=True, verbose=0)
]

t0 = time.time()
historico = modelo.fit(
    X_treino_s, y_ref[idx_treino],
    validation_data=(X_val_s, y_ref[idx_val]),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print(f"Treinamento concluído em {time.time() - t0:.2f} segundos.")
"""),
        code_cell("""# 4. Gráfico 1 — Curvas de Aprendizado (Acurácia e Loss)
fig_aprendizado = plotar_curvas_aprendizado(historico, "Experimento 02: Curvas de Aprendizado MLP (Treinado no Fujita Limpo)")
fig_aprendizado.show()
"""),
        code_cell("""# 5. Avaliação Baseline Limpa (Conjunto de Teste Fujita Holdout 15%)
probs_teste_limpo = modelo.predict(X_teste_limpo_s, verbose=0)
y_pred_teste_limpo = np.argmax(probs_teste_limpo, axis=1)

relatorio_limpo = classification_report(
    y_ref[idx_teste], y_pred_teste_limpo,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO BASELINE: FUJITA TESTE 15% ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_limpo, target_names=NOMES_CLASSES))

fig_cm_limpo = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_limpo,
    "Experimento 02: Matriz de Confusão — Fujita Teste Limpo (Baseline)"
)
fig_cm_limpo.show()
"""),
        code_cell("""# 6. Avaliação Cross-Dataset com Ruído (100% Células Válidas do Mathys)
probs_alvo_ruido = modelo.predict(X_alvo_ruido_s, verbose=0)
y_pred_alvo_ruido = np.argmax(probs_alvo_ruido, axis=1)

relatorio_ruido = classification_report(
    y_alvo, y_pred_alvo_ruido,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO: MATHYS COMPLETO COM RUÍDO 0,5 (INFERÊNCIA FUJITA -> MATHYS) ===")
print(classification_report(y_alvo, y_pred_alvo_ruido, target_names=NOMES_CLASSES))

fig_cm_ruido = plotar_matriz_confusao(
    y_alvo, y_pred_alvo_ruido,
    "Experimento 02: Matriz de Confusão — Mathys Completo com Ruído 0,5"
)
fig_cm_ruido.show()
"""),
        code_cell("""# 7. Gráfico 4 — Comparativo de Desempenho (Fujita Limpo vs. Mathys com Ruído 0,5)
fig_comp = plotar_comparativo(
    relatorio_limpo, relatorio_ruido,
    "Experimento 02 (Fujita -> Mathys): Queda Cross-Dataset com Ruído 0,5",
    nome_cenario_ruido="Mathys com Ruído 0,5"
)
fig_comp.show()

acc_limpo = relatorio_limpo["accuracy"]
acc_ruido = relatorio_ruido["accuracy"]
macro_limpo = relatorio_limpo["macro avg"]["f1-score"]
macro_ruido = relatorio_ruido["macro avg"]["f1-score"]

print("=" * 60)
print(f"Acurácia Global   : {acc_limpo*100:.2f}% (Fujita Limpo) -> {acc_ruido*100:.2f}% (Mathys Ruído) | Delta: {(acc_limpo - acc_ruido)*100:+.2f} p.p.")
print(f"Macro F1-Score    : {macro_limpo*100:.2f}% -> {macro_ruido*100:.2f}% | Delta: {(macro_limpo - macro_ruido)*100:+.2f} p.p.")
print("=" * 60)
""")
    ]
    with open(EXP_DIR / "exp02_fujita_para_mathys.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, indent=1, ensure_ascii=False)
    print("exp02_fujita_para_mathys.ipynb criado.")


# ==============================================================================
# GERADOR DO EXPERIMENTO 03: Mathys -> Mathys (Intra)
# ==============================================================================
def gerar_exp03():
    cells = [
        md_cell("""# Experimento 03: Validação de Sensibilidade ao Ruído (0,5) — Mathys para Mathys (Intra-Dataset)

## Objetivo
Avaliar se a inserção do ruído sentinela neutro (`0,5`) nos genes do **Mathys** que estão ausentes em **Pan** ou **Fujita** degrada a capacidade preditiva do classificador celular **MLP** treinado sobre as projeções limpas do SWeeP 600D.

- **Dataset de Treino (Referência Limpa):** Mathys (`45.572` células válidas, classes 1–7)
- **Dataset de Teste com Ruído:** Mathys com `0,5` nas colunas ausentes na união de ausências (`G_Mathys \\ (G_Pan ∩ G_Fujita ∩ G_Mathys)`)
- **Espaço de Projeção:** `32.738` genes $\\to$ 600D com `orthbase_mathys_600d.npy`
- **Validação:** Teste Holdout (15% sem vazamento) e Dataset Completo (100%)
"""),
        code_cell(HEADER_CODE),
        code_cell(UTILS_CODE),
        code_cell("""# 1. Carregamento dos Dados, Rótulos e Ortobase
PATH_GENES_FUJITA = DADOS_DIR / "fujita" / "featuresF.tsv"
PATH_GENES_PAN = DADOS_DIR / "pan" / "featuresPANcorrigido.tsv"
PATH_GENES_MATHYS = DADOS_DIR / "mathys" / "featuresM.tsv"

genes_fujita = carregar_ensembl_ids(PATH_GENES_FUJITA)
genes_pan = carregar_ensembl_ids(PATH_GENES_PAN)
genes_mathys = carregar_ensembl_ids(PATH_GENES_MATHYS)

s_triple = set(genes_pan) & set(genes_fujita) & set(genes_mathys)
mask_ruido = np.array([g not in s_triple for g in genes_mathys])
print(f"Total genes Mathys : {len(genes_mathys)}")
print(f"Genes na interseção: {len(s_triple)}")
print(f"Genes com ruído 0.5: {mask_ruido.sum()} ({mask_ruido.sum()/len(genes_mathys)*100:.2f}%)")

orthbase_ref = np.load(DADOS_DIR / "mathys" / "orthbase_mathys_600d.npy").astype(np.float32)
X_limpo = np.load(DADOS_DIR / "mathys" / "X_mathys_600d.npy").astype(np.float32)
y_ref = np.load(DADOS_DIR / "mathys" / "y_mathys.npy")
X_ruido = np.load(DADOS_DIR / "mathys" / "X_mathys_ruido_600d.npy").astype(np.float32)

# TRAVAS DE SEGURANÇA (ASSERTIONS FAIL-FAST)
assert orthbase_ref.shape == (len(genes_mathys), DIM_PROJECAO), f"Ortobase inválida: {orthbase_ref.shape}"
assert X_limpo.shape == (len(y_ref), DIM_PROJECAO), f"X_limpo inválido: {X_limpo.shape}"
assert X_ruido.shape == (len(y_ref), DIM_PROJECAO), f"X_ruido inválido: {X_ruido.shape}"
print("Todas as travas de integridade passaram com sucesso!")
"""),
        code_cell("""# 2. Particionamento Estratificado (70% Treino / 15% Validação / 15% Teste Holdout)
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=y_ref
)
razao_val = 0.15 / (1.0 - 0.15)
idx_treino, idx_val = train_test_split(
    idx_treino_val, test_size=razao_val, random_state=SEED, stratify=y_ref[idx_treino_val]
)

print(f"Treino    : {len(idx_treino)} células ({len(idx_treino)/len(indices)*100:.1f}%)")
print(f"Validação : {len(idx_val)} células ({len(idx_val)/len(indices)*100:.1f}%)")
print(f"Teste     : {len(idx_teste)} células ({len(idx_teste)/len(indices)*100:.1f}%)")

scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_limpo[idx_treino])
X_val_s = scaler.transform(X_limpo[idx_val])
X_teste_limpo_s = scaler.transform(X_limpo[idx_teste])

X_teste_ruido_s = scaler.transform(X_ruido[idx_teste])
X_full_ruido_s = scaler.transform(X_ruido)

joblib.dump(scaler, DADOS_DIR / "mathys" / "scaler_exp03.pkl")
"""),
        code_cell("""# 3. Treinamento do Classificador MLP
modelo = construir_modelo_mlp(input_dim=DIM_PROJECAO, n_classes=len(CLASSES_VALIDAS))
caminho_modelo = DADOS_DIR / "mathys" / "modelo_mlp_exp03_mathys_para_mathys.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(caminho_modelo), monitor="val_loss", save_best_only=True, verbose=0)
]

t0 = time.time()
historico = modelo.fit(
    X_treino_s, y_ref[idx_treino],
    validation_data=(X_val_s, y_ref[idx_val]),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print(f"Treinamento concluído em {time.time() - t0:.2f} segundos.")
"""),
        code_cell("""# 4. Gráfico 1 — Curvas de Aprendizado (Acurácia e Loss)
fig_aprendizado = plotar_curvas_aprendizado(historico, "Experimento 03: Curvas de Aprendizado MLP (Mathys Limpo)")
fig_aprendizado.show()
"""),
        code_cell("""# 5. Avaliação Baseline Limpa (Conjunto de Teste Holdout 15%)
probs_teste_limpo = modelo.predict(X_teste_limpo_s, verbose=0)
y_pred_teste_limpo = np.argmax(probs_teste_limpo, axis=1)

relatorio_limpo = classification_report(
    y_ref[idx_teste], y_pred_teste_limpo,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO BASELINE: MATHYS TESTE 15% ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_limpo, target_names=NOMES_CLASSES))

fig_cm_limpo = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_limpo,
    "Experimento 03: Matriz de Confusão — Mathys Teste Limpo (Baseline)"
)
fig_cm_limpo.show()
"""),
        code_cell("""# 6. Avaliação com Ruído Sentinela 0,5 (Teste Holdout 15% e Dataset Completo 100%)
# 6.1 Teste Holdout com Ruído
probs_teste_ruido = modelo.predict(X_teste_ruido_s, verbose=0)
y_pred_teste_ruido = np.argmax(probs_teste_ruido, axis=1)

relatorio_ruido = classification_report(
    y_ref[idx_teste], y_pred_teste_ruido,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO: TESTE HOLDOUT COM RUÍDO 0,5 ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_ruido, target_names=NOMES_CLASSES))

fig_cm_ruido = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_ruido,
    "Experimento 03: Matriz de Confusão — Teste Holdout com Ruído 0,5"
)
fig_cm_ruido.show()

# 6.2 Dataset Completo 100% com Ruído
probs_full_ruido = modelo.predict(X_full_ruido_s, verbose=0)
y_pred_full_ruido = np.argmax(probs_full_ruido, axis=1)
print(f"Acurácia no Dataset Completo (100%) com Ruído: {accuracy_score(y_ref, y_pred_full_ruido):.4f}")
"""),
        code_cell("""# 7. Gráfico 4 — Comparativo de Desempenho (Baseline Limpo vs. Ruído 0,5)
fig_comp = plotar_comparativo(
    relatorio_limpo, relatorio_ruido,
    "Experimento 03 (Mathys -> Mathys): Queda de Desempenho com Ruído 0,5",
    nome_cenario_ruido="Mathys com Ruído 0,5"
)
fig_comp.show()

acc_limpo = relatorio_limpo["accuracy"]
acc_ruido = relatorio_ruido["accuracy"]
macro_limpo = relatorio_limpo["macro avg"]["f1-score"]
macro_ruido = relatorio_ruido["macro avg"]["f1-score"]

print("=" * 60)
print(f"Acurácia Global   : {acc_limpo*100:.2f}% -> {acc_ruido*100:.2f}% (Queda de {(acc_limpo - acc_ruido)*100:+.2f} p.p.)")
print(f"Macro F1-Score    : {macro_limpo*100:.2f}% -> {macro_ruido*100:.2f}% (Queda de {(macro_limpo - macro_ruido)*100:+.2f} p.p.)")
print("=" * 60)
""")
    ]
    with open(EXP_DIR / "exp03_mathys_para_mathys.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, indent=1, ensure_ascii=False)
    print("exp03_mathys_para_mathys.ipynb criado.")


# ==============================================================================
# GERADOR DO EXPERIMENTO 04: Pan -> Pan (Intra)
# ==============================================================================
def gerar_exp04():
    cells = [
        md_cell("""# Experimento 04: Validação de Sensibilidade ao Ruído (0,5) — Pan para Pan (Intra-Dataset)

## Objetivo
Avaliar se a inserção do ruído sentinela neutro (`0,5`) nos genes do **Pan** que estão ausentes em **Fujita** ou **Mathys** degrada a capacidade preditiva do classificador celular **MLP** treinado sobre as projeções limpas do SWeeP 600D.

- **Dataset de Treino (Referência Limpa):** Pan (`57.457` células válidas, classes 1–7)
- **Dataset de Teste com Ruído:** Pan com `0,5` nas colunas ausentes na união de ausências (`G_Pan \\ (G_Pan ∩ G_Fujita ∩ G_Mathys)`)
- **Espaço de Projeção:** `61.541` genes $\\to$ 600D com `orthbase_pan_600d.npy`
- **Validação:** Teste Holdout (15% sem vazamento) e Dataset Completo (100%)
"""),
        code_cell(HEADER_CODE),
        code_cell(UTILS_CODE),
        code_cell("""# 1. Carregamento dos Dados, Rótulos e Ortobase
PATH_GENES_FUJITA = DADOS_DIR / "fujita" / "featuresF.tsv"
PATH_GENES_PAN = DADOS_DIR / "pan" / "featuresPANcorrigido.tsv"
PATH_GENES_MATHYS = DADOS_DIR / "mathys" / "featuresM.tsv"

genes_fujita = carregar_ensembl_ids(PATH_GENES_FUJITA)
genes_pan = carregar_ensembl_ids(PATH_GENES_PAN)
genes_mathys = carregar_ensembl_ids(PATH_GENES_MATHYS)

s_triple = set(genes_pan) & set(genes_fujita) & set(genes_mathys)
mask_ruido = np.array([g not in s_triple for g in genes_pan])
print(f"Total genes Pan    : {len(genes_pan)}")
print(f"Genes na interseção: {len(s_triple)}")
print(f"Genes com ruído 0.5: {mask_ruido.sum()} ({mask_ruido.sum()/len(genes_pan)*100:.2f}%)")

orthbase_ref = np.load(DADOS_DIR / "pan" / "orthbase_pan_600d.npy").astype(np.float32)
X_limpo = np.load(DADOS_DIR / "pan" / "X_pan_600d.npy").astype(np.float32)
y_ref = np.load(DADOS_DIR / "pan" / "y_pan.npy")
X_ruido = np.load(DADOS_DIR / "pan" / "X_pan_ruido_600d.npy").astype(np.float32)

# TRAVAS DE SEGURANÇA (ASSERTIONS FAIL-FAST)
assert orthbase_ref.shape == (len(genes_pan), DIM_PROJECAO), f"Ortobase inválida: {orthbase_ref.shape}"
assert X_limpo.shape == (len(y_ref), DIM_PROJECAO), f"X_limpo inválido: {X_limpo.shape}"
assert X_ruido.shape == (len(y_ref), DIM_PROJECAO), f"X_ruido inválido: {X_ruido.shape}"
print("Todas as travas de integridade passaram com sucesso!")
"""),
        code_cell("""# 2. Particionamento Estratificado (70% Treino / 15% Validação / 15% Teste Holdout)
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=y_ref
)
razao_val = 0.15 / (1.0 - 0.15)
idx_treino, idx_val = train_test_split(
    idx_treino_val, test_size=razao_val, random_state=SEED, stratify=y_ref[idx_treino_val]
)

print(f"Treino    : {len(idx_treino)} células ({len(idx_treino)/len(indices)*100:.1f}%)")
print(f"Validação : {len(idx_val)} células ({len(idx_val)/len(indices)*100:.1f}%)")
print(f"Teste     : {len(idx_teste)} células ({len(idx_teste)/len(indices)*100:.1f}%)")

scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_limpo[idx_treino])
X_val_s = scaler.transform(X_limpo[idx_val])
X_teste_limpo_s = scaler.transform(X_limpo[idx_teste])

X_teste_ruido_s = scaler.transform(X_ruido[idx_teste])
X_full_ruido_s = scaler.transform(X_ruido)

joblib.dump(scaler, DADOS_DIR / "pan" / "scaler_exp04.pkl")
"""),
        code_cell("""# 3. Treinamento do Classificador MLP
modelo = construir_modelo_mlp(input_dim=DIM_PROJECAO, n_classes=len(CLASSES_VALIDAS))
caminho_modelo = DADOS_DIR / "pan" / "modelo_mlp_exp04_pan_para_pan.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(caminho_modelo), monitor="val_loss", save_best_only=True, verbose=0)
]

t0 = time.time()
historico = modelo.fit(
    X_treino_s, y_ref[idx_treino],
    validation_data=(X_val_s, y_ref[idx_val]),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print(f"Treinamento concluído em {time.time() - t0:.2f} segundos.")
"""),
        code_cell("""# 4. Gráfico 1 — Curvas de Aprendizado (Acurácia e Loss)
fig_aprendizado = plotar_curvas_aprendizado(historico, "Experimento 04: Curvas de Aprendizado MLP (Pan Limpo)")
fig_aprendizado.show()
"""),
        code_cell("""# 5. Avaliação Baseline Limpa (Conjunto de Teste Holdout 15%)
probs_teste_limpo = modelo.predict(X_teste_limpo_s, verbose=0)
y_pred_teste_limpo = np.argmax(probs_teste_limpo, axis=1)

relatorio_limpo = classification_report(
    y_ref[idx_teste], y_pred_teste_limpo,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO BASELINE: PAN TESTE 15% ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_limpo, target_names=NOMES_CLASSES))

fig_cm_limpo = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_limpo,
    "Experimento 04: Matriz de Confusão — Pan Teste Limpo (Baseline)"
)
fig_cm_limpo.show()
"""),
        code_cell("""# 6. Avaliação com Ruído Sentinela 0,5 (Teste Holdout 15% e Dataset Completo 100%)
# 6.1 Teste Holdout com Ruído
probs_teste_ruido = modelo.predict(X_teste_ruido_s, verbose=0)
y_pred_teste_ruido = np.argmax(probs_teste_ruido, axis=1)

relatorio_ruido = classification_report(
    y_ref[idx_teste], y_pred_teste_ruido,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO: TESTE HOLDOUT COM RUÍDO 0,5 ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_ruido, target_names=NOMES_CLASSES))

fig_cm_ruido = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_ruido,
    "Experimento 04: Matriz de Confusão — Teste Holdout com Ruído 0,5"
)
fig_cm_ruido.show()

# 6.2 Dataset Completo 100% com Ruído
probs_full_ruido = modelo.predict(X_full_ruido_s, verbose=0)
y_pred_full_ruido = np.argmax(probs_full_ruido, axis=1)
print(f"Acurácia no Dataset Completo (100%) com Ruído: {accuracy_score(y_ref, y_pred_full_ruido):.4f}")
"""),
        code_cell("""# 7. Gráfico 4 — Comparativo de Desempenho (Baseline Limpo vs. Ruído 0,5)
fig_comp = plotar_comparativo(
    relatorio_limpo, relatorio_ruido,
    "Experimento 04 (Pan -> Pan): Queda de Desempenho com Ruído 0,5",
    nome_cenario_ruido="Pan com Ruído 0,5"
)
fig_comp.show()

acc_limpo = relatorio_limpo["accuracy"]
acc_ruido = relatorio_ruido["accuracy"]
macro_limpo = relatorio_limpo["macro avg"]["f1-score"]
macro_ruido = relatorio_ruido["macro avg"]["f1-score"]

print("=" * 60)
print(f"Acurácia Global   : {acc_limpo*100:.2f}% -> {acc_ruido*100:.2f}% (Queda de {(acc_limpo - acc_ruido)*100:+.2f} p.p.)")
print(f"Macro F1-Score    : {macro_limpo*100:.2f}% -> {macro_ruido*100:.2f}% (Queda de {(macro_limpo - macro_ruido)*100:+.2f} p.p.)")
print("=" * 60)
""")
    ]
    with open(EXP_DIR / "exp04_pan_para_pan.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, indent=1, ensure_ascii=False)
    print("exp04_pan_para_pan.ipynb criado.")


# ==============================================================================
# GERADOR DO EXPERIMENTO 05: Pan -> Fujita (Cross)
# ==============================================================================
def gerar_exp05():
    cells = [
        md_cell("""# Experimento 05: Validação de Sensibilidade ao Ruído (0,5) — Pan para Fujita (Cross-Dataset)

## Objetivo
Avaliar a capacidade de generalização e tolerância ao ruído do classificador celular **MLP treinado na referência limpa Pan**, ao realizar inferência sobre o dataset **Fujita**, cujos genes não medidos no Fujita recebem o valor sentinela `0,5` no espaço de colunas do Pan.

- **Dataset de Treino (Referência Limpa):** Pan (`57.457` células válidas, classes 1–7)
- **Dataset de Inferência Alvo:** Fujita (`26.763` células válidas, classes 1–7) alinhado ao Pan
- **Espaço de Projeção:** `61.541` genes $\\to$ 600D com **`orthbase_pan_600d.npy`** (REUSO ESTRITO DA BASE DA REFERÊNCIA)
- **Ruído Injetado:** Genes do Pan ausentes no Fujita recebem `0,5`
"""),
        code_cell(HEADER_CODE),
        code_cell(UTILS_CODE),
        code_cell("""# 1. Carregamento dos Dados, Rótulos e Ortobase da Referência
PATH_GENES_FUJITA = DADOS_DIR / "fujita" / "featuresF.tsv"
PATH_GENES_PAN = DADOS_DIR / "pan" / "featuresPANcorrigido.tsv"

genes_fujita = carregar_ensembl_ids(PATH_GENES_FUJITA)
genes_pan = carregar_ensembl_ids(PATH_GENES_PAN)

s_fujita = set(genes_fujita)
genes_comuns = [g for g in genes_pan if g in s_fujita]
genes_ausentes = [g for g in genes_pan if g not in s_fujita]
print(f"Referência (Pan)    : {len(genes_pan)} genes")
print(f"Alvo (Fujita)       : {len(genes_fujita)} genes")
print(f"Genes em Comum      : {len(genes_comuns)}")
print(f"Genes Ausentes (0.5): {len(genes_ausentes)} ({len(genes_ausentes)/len(genes_pan)*100:.2f}%)")

# Base Ortonormal OBRIGATORIAMENTE da REFERÊNCIA (Pan)
orthbase_ref = np.load(DADOS_DIR / "pan" / "orthbase_pan_600d.npy").astype(np.float32)
X_ref_limpo = np.load(DADOS_DIR / "pan" / "X_pan_600d.npy").astype(np.float32)
y_ref = np.load(DADOS_DIR / "pan" / "y_pan.npy")

# Projeção do Alvo (Fujita) no Espaço da Referência
X_alvo_ruido = np.load(DADOS_DIR / "fujita" / "X_fujita_alinhado_pan_ruido_600d.npy").astype(np.float32)
y_alvo = np.load(DADOS_DIR / "fujita" / "y_fujita.npy")

# TRAVAS DE SEGURANÇA (ASSERTIONS FAIL-FAST)
assert orthbase_ref.shape == (len(genes_pan), DIM_PROJECAO), f"Ortobase inválida: {orthbase_ref.shape}"
assert X_ref_limpo.shape == (len(y_ref), DIM_PROJECAO), f"X_ref inválido: {X_ref_limpo.shape}"
assert X_alvo_ruido.shape == (len(y_alvo), DIM_PROJECAO), f"X_alvo inválido: {X_alvo_ruido.shape}"
print("Travas de segurança e auditoria de ortobase passaram com 100% de sucesso!")
"""),
        code_cell("""# 2. Particionamento e Escalonamento na Referência Limpa (Pan)
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=y_ref
)
razao_val = 0.15 / (1.0 - 0.15)
idx_treino, idx_val = train_test_split(
    idx_treino_val, test_size=razao_val, random_state=SEED, stratify=y_ref[idx_treino_val]
)

scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_ref_limpo[idx_treino])
X_val_s = scaler.transform(X_ref_limpo[idx_val])
X_teste_limpo_s = scaler.transform(X_ref_limpo[idx_teste])

# Escalonamento dos dados do Alvo (Fujita) usando o Scaler da Referência
X_alvo_ruido_s = scaler.transform(X_alvo_ruido)

joblib.dump(scaler, DADOS_DIR / "pan" / "scaler_exp05.pkl")
"""),
        code_cell("""# 3. Treinamento do Classificador MLP na Referência Limpa (Pan)
modelo = construir_modelo_mlp(input_dim=DIM_PROJECAO, n_classes=len(CLASSES_VALIDAS))
caminho_modelo = DADOS_DIR / "pan" / "modelo_mlp_exp05_pan_para_fujita.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(caminho_modelo), monitor="val_loss", save_best_only=True, verbose=0)
]

t0 = time.time()
historico = modelo.fit(
    X_treino_s, y_ref[idx_treino],
    validation_data=(X_val_s, y_ref[idx_val]),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print(f"Treinamento concluído em {time.time() - t0:.2f} segundos.")
"""),
        code_cell("""# 4. Gráfico 1 — Curvas de Aprendizado (Acurácia e Loss)
fig_aprendizado = plotar_curvas_aprendizado(historico, "Experimento 05: Curvas de Aprendizado MLP (Treinado no Pan Limpo)")
fig_aprendizado.show()
"""),
        code_cell("""# 5. Avaliação Baseline Limpa (Conjunto de Teste Pan Holdout 15%)
probs_teste_limpo = modelo.predict(X_teste_limpo_s, verbose=0)
y_pred_teste_limpo = np.argmax(probs_teste_limpo, axis=1)

relatorio_limpo = classification_report(
    y_ref[idx_teste], y_pred_teste_limpo,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO BASELINE: PAN TESTE 15% ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_limpo, target_names=NOMES_CLASSES))

fig_cm_limpo = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_limpo,
    "Experimento 05: Matriz de Confusão — Pan Teste Limpo (Baseline)"
)
fig_cm_limpo.show()
"""),
        code_cell("""# 6. Avaliação Cross-Dataset com Ruído (100% Células Válidas do Fujita)
probs_alvo_ruido = modelo.predict(X_alvo_ruido_s, verbose=0)
y_pred_alvo_ruido = np.argmax(probs_alvo_ruido, axis=1)

relatorio_ruido = classification_report(
    y_alvo, y_pred_alvo_ruido,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO: FUJITA COMPLETO COM RUÍDO 0,5 (INFERÊNCIA PAN -> FUJITA) ===")
print(classification_report(y_alvo, y_pred_alvo_ruido, target_names=NOMES_CLASSES))

fig_cm_ruido = plotar_matriz_confusao(
    y_alvo, y_pred_alvo_ruido,
    "Experimento 05: Matriz de Confusão — Fujita Completo com Ruído 0,5"
)
fig_cm_ruido.show()
"""),
        code_cell("""# 7. Gráfico 4 — Comparativo de Desempenho (Pan Limpo vs. Fujita com Ruído 0,5)
fig_comp = plotar_comparativo(
    relatorio_limpo, relatorio_ruido,
    "Experimento 05 (Pan -> Fujita): Queda Cross-Dataset com Ruído 0,5",
    nome_cenario_ruido="Fujita com Ruído 0,5"
)
fig_comp.show()

acc_limpo = relatorio_limpo["accuracy"]
acc_ruido = relatorio_ruido["accuracy"]
macro_limpo = relatorio_limpo["macro avg"]["f1-score"]
macro_ruido = relatorio_ruido["macro avg"]["f1-score"]

print("=" * 60)
print(f"Acurácia Global   : {acc_limpo*100:.2f}% (Pan Limpo) -> {acc_ruido*100:.2f}% (Fujita Ruído) | Delta: {(acc_limpo - acc_ruido)*100:+.2f} p.p.")
print(f"Macro F1-Score    : {macro_limpo*100:.2f}% -> {macro_ruido*100:.2f}% | Delta: {(macro_limpo - macro_ruido)*100:+.2f} p.p.")
print("=" * 60)
""")
    ]
    with open(EXP_DIR / "exp05_pan_para_fujita.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, indent=1, ensure_ascii=False)
    print("exp05_pan_para_fujita.ipynb criado.")


# ==============================================================================
# GERADOR DO EXPERIMENTO 06: Pan -> Mathys (Cross)
# ==============================================================================
def gerar_exp06():
    cells = [
        md_cell("""# Experimento 06: Validação de Sensibilidade ao Ruído (0,5) — Pan para Mathys (Cross-Dataset)

## Objetivo
Avaliar a capacidade de generalização e tolerância ao ruído do classificador celular **MLP treinado na referência limpa Pan**, ao realizar inferência sobre o dataset **Mathys**, cujos genes não medidos no Mathys recebem o valor sentinela `0,5` no espaço de colunas do Pan.

- **Dataset de Treino (Referência Limpa):** Pan (`57.457` células válidas, classes 1–7)
- **Dataset de Inferência Alvo:** Mathys (`45.572` células válidas, classes 1–7) alinhado ao Pan
- **Espaço de Projeção:** `61.541` genes $\\to$ 600D com **`orthbase_pan_600d.npy`** (REUSO ESTRITO DA BASE DA REFERÊNCIA)
- **Ruído Injetado:** Genes do Pan ausentes no Mathys recebem `0,5`
"""),
        code_cell(HEADER_CODE),
        code_cell(UTILS_CODE),
        code_cell("""# 1. Carregamento dos Dados, Rótulos e Ortobase da Referência
PATH_GENES_MATHYS = DADOS_DIR / "mathys" / "featuresM.tsv"
PATH_GENES_PAN = DADOS_DIR / "pan" / "featuresPANcorrigido.tsv"

genes_mathys = carregar_ensembl_ids(PATH_GENES_MATHYS)
genes_pan = carregar_ensembl_ids(PATH_GENES_PAN)

s_mathys = set(genes_mathys)
genes_comuns = [g for g in genes_pan if g in s_mathys]
genes_ausentes = [g for g in genes_pan if g not in s_mathys]
print(f"Referência (Pan)    : {len(genes_pan)} genes")
print(f"Alvo (Mathys)       : {len(genes_mathys)} genes")
print(f"Genes em Comum      : {len(genes_comuns)}")
print(f"Genes Ausentes (0.5): {len(genes_ausentes)} ({len(genes_ausentes)/len(genes_pan)*100:.2f}%)")

# Base Ortonormal OBRIGATORIAMENTE da REFERÊNCIA (Pan)
orthbase_ref = np.load(DADOS_DIR / "pan" / "orthbase_pan_600d.npy").astype(np.float32)
X_ref_limpo = np.load(DADOS_DIR / "pan" / "X_pan_600d.npy").astype(np.float32)
y_ref = np.load(DADOS_DIR / "pan" / "y_pan.npy")

# Projeção do Alvo (Mathys) no Espaço da Referência
X_alvo_ruido = np.load(DADOS_DIR / "mathys" / "X_mathys_alinhado_pan_ruido_600d.npy").astype(np.float32)
y_alvo = np.load(DADOS_DIR / "mathys" / "y_mathys.npy")

# TRAVAS DE SEGURANÇA (ASSERTIONS FAIL-FAST)
assert orthbase_ref.shape == (len(genes_pan), DIM_PROJECAO), f"Ortobase inválida: {orthbase_ref.shape}"
assert X_ref_limpo.shape == (len(y_ref), DIM_PROJECAO), f"X_ref inválido: {X_ref_limpo.shape}"
assert X_alvo_ruido.shape == (len(y_alvo), DIM_PROJECAO), f"X_alvo inválido: {X_alvo_ruido.shape}"
print("Travas de segurança e auditoria de ortobase passaram com 100% de sucesso!")
"""),
        code_cell("""# 2. Particionamento e Escalonamento na Referência Limpa (Pan)
indices = np.arange(len(y_ref))
idx_treino_val, idx_teste = train_test_split(
    indices, test_size=0.15, random_state=SEED, stratify=y_ref
)
razao_val = 0.15 / (1.0 - 0.15)
idx_treino, idx_val = train_test_split(
    idx_treino_val, test_size=razao_val, random_state=SEED, stratify=y_ref[idx_treino_val]
)

scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_ref_limpo[idx_treino])
X_val_s = scaler.transform(X_ref_limpo[idx_val])
X_teste_limpo_s = scaler.transform(X_ref_limpo[idx_teste])

# Escalonamento dos dados do Alvo (Mathys) usando o Scaler da Referência
X_alvo_ruido_s = scaler.transform(X_alvo_ruido)

joblib.dump(scaler, DADOS_DIR / "pan" / "scaler_exp06.pkl")
"""),
        code_cell("""# 3. Treinamento do Classificador MLP na Referência Limpa (Pan)
modelo = construir_modelo_mlp(input_dim=DIM_PROJECAO, n_classes=len(CLASSES_VALIDAS))
caminho_modelo = DADOS_DIR / "pan" / "modelo_mlp_exp06_pan_para_mathys.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(caminho_modelo), monitor="val_loss", save_best_only=True, verbose=0)
]

t0 = time.time()
historico = modelo.fit(
    X_treino_s, y_ref[idx_treino],
    validation_data=(X_val_s, y_ref[idx_val]),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print(f"Treinamento concluído em {time.time() - t0:.2f} segundos.")
"""),
        code_cell("""# 4. Gráfico 1 — Curvas de Aprendizado (Acurácia e Loss)
fig_aprendizado = plotar_curvas_aprendizado(historico, "Experimento 06: Curvas de Aprendizado MLP (Treinado no Pan Limpo)")
fig_aprendizado.show()
"""),
        code_cell("""# 5. Avaliação Baseline Limpa (Conjunto de Teste Pan Holdout 15%)
probs_teste_limpo = modelo.predict(X_teste_limpo_s, verbose=0)
y_pred_teste_limpo = np.argmax(probs_teste_limpo, axis=1)

relatorio_limpo = classification_report(
    y_ref[idx_teste], y_pred_teste_limpo,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO BASELINE: PAN TESTE 15% ===")
print(classification_report(y_ref[idx_teste], y_pred_teste_limpo, target_names=NOMES_CLASSES))

fig_cm_limpo = plotar_matriz_confusao(
    y_ref[idx_teste], y_pred_teste_limpo,
    "Experimento 06: Matriz de Confusão — Pan Teste Limpo (Baseline)"
)
fig_cm_limpo.show()
"""),
        code_cell("""# 6. Avaliação Cross-Dataset com Ruído (100% Células Válidas do Mathys)
probs_alvo_ruido = modelo.predict(X_alvo_ruido_s, verbose=0)
y_pred_alvo_ruido = np.argmax(probs_alvo_ruido, axis=1)

relatorio_ruido = classification_report(
    y_alvo, y_pred_alvo_ruido,
    target_names=NOMES_CLASSES,
    output_dict=True
)
print("=== RELATÓRIO: MATHYS COMPLETO COM RUÍDO 0,5 (INFERÊNCIA PAN -> MATHYS) ===")
print(classification_report(y_alvo, y_pred_alvo_ruido, target_names=NOMES_CLASSES))

fig_cm_ruido = plotar_matriz_confusao(
    y_alvo, y_pred_alvo_ruido,
    "Experimento 06: Matriz de Confusão — Mathys Completo com Ruído 0,5"
)
fig_cm_ruido.show()
"""),
        code_cell("""# 7. Gráfico 4 — Comparativo de Desempenho (Pan Limpo vs. Mathys com Ruído 0,5)
fig_comp = plotar_comparativo(
    relatorio_limpo, relatorio_ruido,
    "Experimento 06 (Pan -> Mathys): Queda Cross-Dataset com Ruído 0,5",
    nome_cenario_ruido="Mathys com Ruído 0,5"
)
fig_comp.show()

acc_limpo = relatorio_limpo["accuracy"]
acc_ruido = relatorio_ruido["accuracy"]
macro_limpo = relatorio_limpo["macro avg"]["f1-score"]
macro_ruido = relatorio_ruido["macro avg"]["f1-score"]

print("=" * 60)
print(f"Acurácia Global   : {acc_limpo*100:.2f}% (Pan Limpo) -> {acc_ruido*100:.2f}% (Mathys Ruído) | Delta: {(acc_limpo - acc_ruido)*100:+.2f} p.p.")
print(f"Macro F1-Score    : {macro_limpo*100:.2f}% -> {macro_ruido*100:.2f}% | Delta: {(macro_limpo - macro_ruido)*100:+.2f} p.p.")
print("=" * 60)
""")
    ]
    with open(EXP_DIR / "exp06_pan_para_mathys.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, indent=1, ensure_ascii=False)
    print("exp06_pan_para_mathys.ipynb criado.")

if __name__ == "__main__":
    gerar_exp01()
    gerar_exp02()
    gerar_exp03()
    gerar_exp04()
    gerar_exp05()
    gerar_exp06()
    print("Todos os 6 notebooks foram gerados com sucesso em experimentos/!")
