# 🤖 Diretrizes para Agentes de IA — Uso Obrigatório do SWeeP Oficial

Este documento estabelece as regras mandatórias para qualquer agente de IA operando neste repositório em tarefas de vetorização, projeção dimensional e processamento de dados biológicos.

---

## 🚫 1. Regra de Ouro: Proibição de Reescrita Algorítmica

> **REGRA MANDATÓRIA:** É **terminantemente proibido** reimplementar, reescrever do zero ou derivar manualmente o algoritmo **SWeeP** (*Spaced Words Projection*) ou suas rotinas de ortobase/projeção no código do projeto.

### Racional Científico:
- Reescrever o algoritmo introduz incerteza e insegurança metodológica para os pesquisadores, que precisam de garantia absoluta de que a transformação matemática é **100% fidedigna e reprodutível** em relação aos métodos oficiais publicados pela Universidade Federal do Paraná (UFPR).
- Sempre utilize as **distribuições oficiais homologadas** mantidas pelos autores da metodologia.

---

## 📦 2. Bibliotecas Oficiais Homologadas

| Ecossistema | Pacote Oficial | Origem / Repositório | Referência de Documentação |
| :--- | :--- | :--- | :--- |
| **Python** | `sweep` (v2.1+) | [PyPI - sweep](https://pypi.org/project/sweep/) | [GitHub - diogomachado-bioinfo/sweep](https://github.com/diogomachado-bioinfo/sweep) |
| **R** | `rSWeeP` (v1.8+) | [Bioconductor - rSWeeP](https://bioconductor.org/packages/rSWeeP) | [GitHub - CamilaPPerico/rSWeeP](https://github.com/CamilaPPerico/rSWeeP) • [Docs AIBIA Lab](https://aibialab.github.io/rSWeeP) |

> **Artigo Científico Base:**  
> De Pierri, C. R., et al. (2020). *SWeeP: representing large biological sequences datasets in compact vectors*. **Nature Scientific Reports**, 10(1), 91. [DOI: 10.1038/s41598-019-55627-4](https://doi.org/10.1038/s41598-019-55627-4).

---

## 🧭 3. Ordem de Precedência Operacional

Sempre que uma solicitação de vetorização ou redução dimensional com SWeeP for realizada, o agente deve seguir rigorosamente a seguinte cadeia de decisão:

1. **Prioridade 1 — Pacote Python Oficial (`sweep`):**
   - Utilizar a biblioteca `sweep` diretamente no ambiente virtual Python gerenciado via `uv`.
   - Utilizar para vetorização de sequências biológicas em formato FASTA (`fas2sweep`) e geração/auditoria de bases ortonormais estritas via SVD (`orthbase`, `is_orthonormal`).
   - *Atenção:* A versão atual do pacote Python oficial (2.1+) não implementa redução de dimensionalidade sobre matrizes de contagens/expressão gênica (scRNA-seq). Para dados matriciais celulares, deve-se adotar obrigatoriamente a Prioridade 2.

2. **Prioridade 2 — Pacote R Oficial (`rSWeeP` v1.18+) via Subprocesso:**
   - Para redução dimensional de matrizes de expressão gênica (scRNA-seq, contagens celulares, `dgCMatrix`), utilize a implementação oficial do `rSWeeP` mantida pelo laboratório AIBIA/UFPR.
   - O agente **não deve** tentar reimplementar a projeção ou aplicar multiplicação matricial solta em Python; deve invocar o script homologado `projetar_rsweep_oficial.R` via `subprocess.run(["Rscript", ...])`, o qual aciona as rotinas oficiais `rSWeeP::orthBase` e `rSWeeP::SWeeP(input, orthbase)`.

---

## 🛠️ 4. Protocolo de Instalação Automatizada

Caso as dependências não estejam presentes no ambiente no momento da execução, o agente deve instalá-las imediatamente:

### No Ambiente Python (`uv`):
```bash
uv add sweep
```

### No Ambiente R (Bioconductor / GitHub Oficial v1.18+):
Para garantir suporte nativo à projeção de matrizes de RNA-seq e integração binária direta com NumPy:
```bash
Rscript -e '
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org");
if (!requireNamespace("Biostrings", quietly = TRUE)) BiocManager::install("Biostrings", ask=FALSE);
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos="https://cloud.r-project.org");
if (!requireNamespace("RcppCNPy", quietly = TRUE)) install.packages("RcppCNPy", repos="https://cloud.r-project.org");
if (!requireNamespace("rSWeeP", quietly = TRUE) || packageVersion("rSWeeP") < "1.18.0") {
    remotes::install_github("CamilaPPerico/rSWeeP", upgrade=FALSE)
}
'
```

---

## 💻 5. Guia Prático de Implementação (Snippets Oficiais)

### Padrão 1: Vetorização de Sequências FASTA e Ortobase com Python (`sweep`)
Para sequências biológicas FASTA ou geração de bases ortonormais SVD:

```python
from sweep import fas2sweep, is_orthonormal, orthbase
import numpy as np

# A. Geração e verificação de ortobase pura (SVD)
orth_mat = orthbase(lin=36601, col=600, seed=42, dtype=np.float32)
assert is_orthonormal(orth_mat.astype(np.float64)), "Base gerada não é estritamente ortonormal!"

# B. Vetorização oficial de sequências FASTA (aminoácidos ou nucleotídeos)
# vetores_compactos = fas2sweep("sequencias.fasta", orth_mat=orth_mat)
```

---

### Padrão 2: Execução do `rSWeeP` em R via Subprocesso Python
Para matrizes de contagens celulares e scRNA-seq, utilize o script homologado `projetar_rsweep_oficial.R`:

```python
import subprocess
from pathlib import Path
import scipy.io as sio

# 1. Exporta matriz esparsa temporária
path_mtx_temp = Path("dados/X_bin_temp.mtx")
sio.mmwrite(path_mtx_temp, X_bin)

# 2. Executa a projeção oficial com rSWeeP::SWeeP
cmd = [
    "Rscript",
    "projetar_rsweep_oficial.R",
    str(path_mtx_temp),
    "dados/orthbase_fujita_600d.npy",
    "dados/X_fujita_600d.npy",
    "600",  # dimensões
    "42",   # seed
    "TRUE"  # remove .mtx após projeção
]
resultado = subprocess.run(cmd, capture_output=True, text=True, check=True)
print(resultado.stdout)

# 3. Recarrega o array oficial projetado
X_projetado = np.load("dados/X_fujita_600d.npy")
```

---

## ✅ Checklist de Conformidade para Agentes

Antes de concluir qualquer tarefa envolvendo SWeeP:
- [ ] O código utiliza `from sweep import ...` ou chamada a `rSWeeP` via `Rscript`?
- [ ] O algoritmo foi importado de pacote oficial e **não** reimplementado manualmente?
- [ ] Se uma nova dependência foi necessária, ela foi registrada no `pyproject.toml` via `uv add`?
- [ ] A semente e parâmetros dimensionais estão registrados para reprodutibilidade?
