"""Pacote de Memória Associativa Moderna (Modern Hopfield Network) acelerada em GPU.

Implementa arquitetura Dense Associative Memory (Ramsauer et al., 2020) com
otimização Bayesiana via KerasTuner, extração de protótipos SWeeP e pipeline
de imputação de genes faltantes em scRNA-seq.
"""

from __future__ import annotations

from .avaliador_hopfield import AvaliadorHopfield
from .estrategias_clusterizacao import (
    EstrategiaClusterizacao,
    EstrategiaKMeansDinamico,
    EstrategiaKMeansFixo,
)
from .extrator_padroes import ExtratorPadroesSubcluster
from .hopfield import ModernHopfieldNetwork
from .hopfield_kerastuner import HopfieldKerasTuner
from .hopfield_utils import closervects, contaocorr, princomp_, sorti, wsort
from .pipeline_imputacao import (
    CLASSES_VALIDAS,
    NOMES_CLASSES,
    anexar_secao_pos_hopfield_notebook,
    calibrar_limiar_otimo_pr,
    carregar_ensembl_ids,
    carregar_matriz_binaria_csr,
    imputar_e_projetar_sweep_streaming,
    obter_particoes_canonicas,
)

__all__ = [
    "ModernHopfieldNetwork",
    "HopfieldKerasTuner",
    "ExtratorPadroesSubcluster",
    "EstrategiaClusterizacao",
    "EstrategiaKMeansFixo",
    "EstrategiaKMeansDinamico",
    "AvaliadorHopfield",
    "closervects",
    "sorti",
    "princomp_",
    "contaocorr",
    "wsort",
    "CLASSES_VALIDAS",
    "NOMES_CLASSES",
    "carregar_ensembl_ids",
    "carregar_matriz_binaria_csr",
    "obter_particoes_canonicas",
    "calibrar_limiar_otimo_pr",
    "imputar_e_projetar_sweep_streaming",
    "anexar_secao_pos_hopfield_notebook",
]
