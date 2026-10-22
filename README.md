# Teste MLP Fujita: Pipeline de Classificação Celular e Imputação

Pipeline unificado para treinamento, avaliação e diagnóstico de modelos MLP em dados transcriptômicos de células únicas (Fujita), incluindo análise de genes faltantes com sentinelas e recuperação pós-imputação via Rede de Hopfield.

---

## 🚀 Como Configurar o Ambiente com `uv`

Este repositório utiliza o gerenciador ultrarrápido [`uv`](https://github.com/astral-sh/uv) com arquivo padrão `pyproject.toml` e lockfile `uv.lock`.

### 1. Pré-requisito: Instalar o `uv`

Caso ainda não possua o `uv` instalado:

- **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Criar o Ambiente Virtual e Sincronizar Dependências

No diretório raiz do projeto, execute:

```bash
uv sync
```

O `uv` irá:
1. Identificar o Python compatível (`>=3.11, <3.13` — definido em `.python-version` como 3.11 para suporte estável ao TensorFlow/Keras).
2. Criar automaticamente a pasta `.venv/`.
3. Instalar todas as dependências travadas no `uv.lock` em poucos segundos.

### 3. Registrar o Kernel Jupyter (Opcional, já pronto para VS Code e Jupyter)

Se desejar registrar explicitamente o kernel para o Jupyter:

```bash
uv run python -m ipykernel install --user --name teste-mlp-fujita --display-name "Python (teste-mlp-fujita)"
```

### 4. Executar o Jupyter Notebook ou JupyterLab

Você pode rodar diretamente via `uv run`:

```bash
uv run jupyter lab
```

Ou ativar o ambiente virtual:

- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

---

## 📦 Pacotes Principais

| Pacote | Função no Projeto |
| :--- | :--- |
| **`tensorflow` / `keras`** | Definição da arquitetura MLP (128 $\to$ 64 $\to$ 32 $\to$ 7), treinamento, extração de embeddings e carregamento do modelo `clasificador_pan.keras` |
| **`scikit-learn`** (1.6.1) | Pré-processamento (`StandardScaler`), particionamento estratificado, métricas (`classification_report`, `ConfusionMatrixDisplay`, `accuracy_score`) e PCA |
| **`umap-learn`** | Projeção não linear 2D do espaço latente 32D para diagnóstico de *Data Drift* / *Batch Effect* |
| **`polars`** | Leitura ultrarrápida e manipulação dos arquivos tabulares de dados brutos e rótulos |
| **`numpy` & `matplotlib`** | Operações matriciais e geração dos gráficos de matriz de confusão e métricas agrupadas |
| **`joblib`** | Serialização e deserialização do normalizador `scaler_celular_pan.pkl` |
| **`anndata` & `h5py`** | Manipulação de matrizes de contagem de células únicas em formato `.h5ad` (`dados/`) |
| **`ipykernel` & `jupyterlab`** | Suporte completo à execução interativa de notebooks |

---

## ⚠️ Observações para Desenvolvedores

1. **Caminhos nos Dados:**
   - No notebook `pipeline_unificado_fujita_fujitaFaltante.ipynb`, as células 5, 29 e 35 contêm referências a caminhos absolutos do Windows (`C:\Users\Leticia\...`). Ao rodar em outra máquina ou no Linux, altere para os caminhos relativos ou arquivos locais correspondentes no diretório `dados/`.
2. **Artefatos Locais:**
   - `clasificador_pan.keras`: Modelo MLP já treinado pronto para inferência.
   - `scaler_celular_pan.pkl`: Instância do `StandardScaler` ajustada no conjunto de treino.
3. **Import Residual:**
   - Na célula 5 do notebook existe uma linha `from sqlalchemy import false` (gerada por auto-import do editor). O pacote `sqlalchemy` foi incluído nas dependências para evitar falhas imediatas, mas essa linha pode ser removida com segurança.
