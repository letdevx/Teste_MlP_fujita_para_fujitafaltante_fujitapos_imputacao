"""Módulo de Otimização de Hiperparâmetros da Rede Hopfield via KerasTuner.

Integra o KerasTuner (Otimização Bayesiana e Random Search) com a
ModernHopfieldNetwork em PyTorch, utilizando Cache LRU de protótipos SWeeP
para garantir execução rápida, escalável e OOM-Safe em scRNA-seq.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal

import keras_tuner as kt  # isort: skip
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .avaliador_hopfield import AvaliadorHopfield
from .estrategias_clusterizacao import (
    EstrategiaKMeansDinamico,
    EstrategiaKMeansFixo,
)
from .extrator_padroes import ExtratorPadroesSubcluster
from .hopfield import ModernHopfieldNetwork

PathType = str | os.PathLike[str]


class HopfieldKerasTuner(kt.Tuner):
    """Tuner customizado KerasTuner para ModernHopfieldNetwork em PyTorch.

    Sobrescreve `run_trial` para gerenciar a extração de protótipos de
    expressão no espaço latente rSWeeP e a avaliação da memória associativa
    Hopfield no conjunto de validação, reutilizando centróides em cache LRU.

    Parameters
    ----------
    w0 : NDArray | sp.spmatrix
        Matriz de expressão binarizada do conjunto de treino (n_células × n_genes).
    wswp : NDArray
        Matriz de projeção compacta rSWeeP 600D do conjunto de treino.
    labels : NDArray | Sequence[int]
        Vetor com os rótulos biológicos canônicos das células de treino.
    x_val : NDArray | sp.spmatrix
        Matriz de expressão do conjunto de validação holdout não-visto.
    y_val : NDArray | Sequence[int]
        Rótulos biológicos das células de validação.
    classes : Sequence[int] | None, optional
        Lista de identificadores das classes canônicas ativas.
    max_cache_padroes : int, default=50
        Capacidade máxima do cache LRU para protótipos SWeeP.
    algorithm : {"bayesian", "random"}, default="bayesian"
        Algoritmo de exploração de hiperparâmetros.
    objective : str, default="val_f1_macro"
        Métrica alvo a ser maximizada.
    max_trials : int, default=20
        Número máximo de configurações a explorar.
    directory : str, default="outputs/kt_hopfield"
        Diretório para persistência de logs e trials do KerasTuner.
    project_name : str, default="hopfield_tuning"
        Nome do projeto/estudo no KerasTuner.
    overwrite : bool, default=True
        Se True, reinicia os trials sobrescrevendo execuções prévias.
    seed : int, default=42
        Semente pseudoaleatória para reprodutibilidade.
    """

    def __init__(
        self,
        w0: NDArray[Any] | sp.spmatrix,
        wswp: NDArray[Any],
        labels: NDArray[Any] | Sequence[int],
        x_val: NDArray[Any] | sp.spmatrix,
        y_val: NDArray[Any] | Sequence[int],
        classes: Sequence[int] | None = None,
        max_cache_padroes: int = 50,
        algorithm: Literal["bayesian", "random"] = "bayesian",
        objective: str = "val_f1_macro",
        max_trials: int = 20,
        directory: str = "outputs/kt_hopfield",
        project_name: str = "hopfield_tuning",
        overwrite: bool = True,
        seed: int = 42,
    ) -> None:
        self.w0_raw: NDArray[np.float32] | sp.csr_matrix
        if sp.issparse(w0):
            self.w0_raw = sp.csr_matrix(w0)
        else:
            self.w0_raw = np.asarray(w0, dtype=np.float32)

        self.wswp: NDArray[np.float32] = np.asarray(wswp, dtype=np.float32)
        self.labels: NDArray[np.int_] = np.asarray(labels, dtype=int)
        self.classes: list[int] = (
            list(classes)
            if classes is not None
            else sorted(np.unique(self.labels).tolist())
        )

        self.x_val: NDArray[np.float32] | sp.csr_matrix
        if sp.issparse(x_val):
            self.x_val = sp.csr_matrix(x_val)
        else:
            self.x_val = np.asarray(x_val, dtype=np.float32)

        self.y_val: NDArray[np.int_] = np.asarray(y_val, dtype=int)
        self.seed: int = int(seed)
        self.max_cache_padroes: int = int(max_cache_padroes)

        # Cache LRU para centróides e protótipos SWeeP
        self._cache_padroes: OrderedDict[
            tuple[int, int, str], tuple[NDArray[np.float32], list[tuple[int, int]]]
        ] = OrderedDict()
        self.estatisticas_cache: dict[str, int] = {"hits": 0, "misses": 0}
        self.historico_trials: list[dict[str, Any]] = []

        # Configuração do oráculo estatístico KerasTuner
        obj_inst = kt.Objective(objective, direction="max")
        oracle: kt.engine.oracle.Oracle
        if algorithm == "bayesian":
            oracle = kt.oracles.BayesianOptimizationOracle(
                objective=obj_inst,
                max_trials=max_trials,
                seed=seed,
            )
        elif algorithm == "random":
            oracle = kt.oracles.RandomSearchOracle(
                objective=obj_inst,
                max_trials=max_trials,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Algoritmo '{algorithm}' não suportado. Escolha 'bayesian' ou 'random'."
            )

        super().__init__(
            oracle=oracle,
            directory=directory,
            project_name=project_name,
            overwrite=overwrite,
        )

    def obter_padroes(
        self,
        nc: int,
        k_vizinhos: int,
        estrategia: Literal["kmeans_fixo", "kmeans_dinamico"],
    ) -> tuple[NDArray[np.float32], list[tuple[int, int]]]:
        """Recupera protótipos de memória do cache LRU ou executa o agrupamento SWeeP.

        Parameters
        ----------
        nc : int
            Número de protótipos/centróides por classe biológica.
        k_vizinhos : int
            Contagem de vizinhos para consenso local do centróide.
        estrategia : {"kmeans_fixo", "kmeans_dinamico"}
            Estratégia de particionamento no espaço latente.

        Returns
        -------
        tuple[NDArray[np.float32], list[tuple[int, int]]]
            Matriz de protótipos representativos e metadados estruturais.
        """
        chave = (int(nc), int(k_vizinhos), str(estrategia))
        if chave in self._cache_padroes:
            self.estatisticas_cache["hits"] += 1
            self._cache_padroes.move_to_end(chave)
            return self._cache_padroes[chave]

        self.estatisticas_cache["misses"] += 1
        estrat_inst = (
            EstrategiaKMeansFixo(n_clusters=nc, seed=self.seed)
            if estrategia == "kmeans_fixo"
            else EstrategiaKMeansDinamico(k_range=[nc], seed=self.seed)
        )

        extrator = ExtratorPadroesSubcluster(
            W0=self.w0_raw,
            labels=self.labels,
            classes=self.classes,
            estrategia=estrat_inst,
            seed=self.seed,
            k=k_vizinhos,
            nc=nc,
        )
        extrator.extrair(self.wswp)

        assert extrator.padroes is not None and extrator.meta is not None
        padroes = extrator.padroes.astype(np.float32)
        meta = list(extrator.meta)

        # Evicção LRU se exceder o teto do cache
        if len(self._cache_padroes) >= self.max_cache_padroes:
            self._cache_padroes.popitem(last=False)

        self._cache_padroes[chave] = (padroes, meta)
        return padroes, meta

    def run_trial(self, trial: kt.engine.trial.Trial, *args: Any, **kwargs: Any) -> Any:
        """Executa um trial de avaliação da ModernHopfieldNetwork no KerasTuner.

        Parameters
        ----------
        trial : kt.engine.trial.Trial
            Objeto do trial contendo os hiperparâmetros sugeridos pelo oráculo.
        *args : Any
            Argumentos adicionais encaminhados pelo tuner.
        **kwargs : Any
            Palavras-chave adicionais.

        Returns
        -------
        dict[str, float]
            Métricas de avaliação registradas para o oráculo.
        """
        hp = trial.hyperparameters

        # Definição do espaço de busca
        nc: int = int(hp.Int("nc", min_value=10, max_value=35, step=5, default=20))
        k_vizinhos: int = int(
            hp.Int("k_vizinhos", min_value=1, max_value=7, step=2, default=3)
        )
        beta: float = float(
            hp.Float("beta", min_value=0.2, max_value=4.0, step=0.2, default=1.0)
        )
        threshold: float = float(
            hp.Float("threshold", min_value=0.1, max_value=0.7, step=0.1, default=0.5)
        )
        normalize: bool = bool(hp.Boolean("normalize", default=False))
        scale_by_dim: bool = bool(hp.Boolean("scale_by_dim", default=True))
        estrategia_raw = hp.Choice(
            "estrategia", ["kmeans_fixo", "kmeans_dinamico"], default="kmeans_fixo"
        )
        estrategia: Literal["kmeans_fixo", "kmeans_dinamico"] = (
            "kmeans_fixo" if estrategia_raw == "kmeans_fixo" else "kmeans_dinamico"
        )

        # 1. Recuperação/extração de protótipos do treino com cache
        padroes, meta = self.obter_padroes(
            nc=nc,
            k_vizinhos=k_vizinhos,
            estrategia=estrategia,
        )

        # 2. Configuração e armazenamento na memória Hopfield
        rede = ModernHopfieldNetwork(
            beta=beta,
            threshold=threshold,
            normalize=normalize,
            scale_by_dim=scale_by_dim,
            binary=True,
        )
        rede.store(padroes)

        # 3. Inferência OOM-Safe em lote sobre o conjunto de validação
        att_weights = rede.compute_attention_weights(
            self.x_val,
            batch_size=2048,
        )

        # 4. Avaliação via Softmax Class Pooling
        avaliador = AvaliadorHopfield(
            padroes=padroes,
            classes=self.classes,
            nc=nc,
            meta=meta,
        )
        avaliador.avaliar_por_atencao(att_weights, self.y_val)

        val_f1_macro = float(
            avaliador.f1_macro if avaliador.f1_macro is not None else 0.0
        )
        val_acuracia = float(
            avaliador.acuracia if avaliador.acuracia is not None else 0.0
        )
        val_f1_weighted = float(
            avaliador.f1_weighted if avaliador.f1_weighted is not None else 0.0
        )

        # Registro no histórico do tuner
        trial_record: dict[str, Any] = {
            "trial_id": trial.trial_id,
            "hiperparametros": {
                "nc": nc,
                "k_vizinhos": k_vizinhos,
                "beta": beta,
                "threshold": threshold,
                "normalize": normalize,
                "scale_by_dim": scale_by_dim,
                "estrategia": estrategia,
            },
            "metricas": {
                "val_f1_macro": val_f1_macro,
                "val_acuracia": val_acuracia,
                "val_f1_weighted": val_f1_weighted,
            },
        }
        self.historico_trials.append(trial_record)

        print(
            f"[Trial {trial.trial_id}] F1 Macro: {val_f1_macro:.4f} | "
            f"Acurácia: {val_acuracia:.4f} | "
            f"nc={nc}, beta={beta:.2f}, k={k_vizinhos}, thresh={threshold:.2f}"
        )

        return {
            "val_f1_macro": val_f1_macro,
            "val_acuracia": val_acuracia,
            "val_f1_weighted": val_f1_weighted,
        }

    def obter_melhor_configuracao(self) -> dict[str, Any]:
        """Retorna o dicionário de hiperparâmetros da configuração campeã.

        Returns
        -------
        dict[str, Any]
            Dicionário com os valores ótimos encontrados.
        """
        best_hp = self.get_best_hyperparameters(1)[0]
        return {
            "nc": int(best_hp.get("nc")),
            "k_vizinhos": int(best_hp.get("k_vizinhos")),
            "beta": float(best_hp.get("beta")),
            "threshold": float(best_hp.get("threshold")),
            "normalize": bool(best_hp.get("normalize")),
            "scale_by_dim": bool(best_hp.get("scale_by_dim")),
            "estrategia": str(best_hp.get("estrategia")),
        }

    def construir_melhor_modelo(
        self,
    ) -> tuple[
        ModernHopfieldNetwork,
        NDArray[np.float32],
        list[tuple[int, int]],
        dict[str, Any],
    ]:
        """Materializa e armazena a rede Hopfield com a configuração campeã.

        Returns
        -------
        tuple[ModernHopfieldNetwork, NDArray[np.float32], list[tuple[int, int]], dict[str, Any]]
            Tupla contendo a instância treinada, a matriz de protótipos, a lista
            de metadados e o dicionário com a configuração campeã.
        """
        cfg = self.obter_melhor_configuracao()
        padroes, meta = self.obter_padroes(
            nc=cfg["nc"],
            k_vizinhos=cfg["k_vizinhos"],
            estrategia=cfg["estrategia"],  # type: ignore[arg-type]
        )

        rede = ModernHopfieldNetwork(
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            normalize=cfg["normalize"],
            scale_by_dim=cfg["scale_by_dim"],
            binary=True,
        )
        rede.store(padroes)
        return rede, padroes, meta, cfg

    def exportar_historico_json(self, path_json: PathType) -> None:
        """Exporta os trials e a configuração campeã em formato JSON.

        Parameters
        ----------
        path_json : str | os.PathLike[str]
            Caminho do arquivo de destino.
        """
        path_str = str(path_json)
        os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
        resumo = {
            "melhor_configuracao": self.obter_melhor_configuracao(),
            "estatisticas_cache": self.estatisticas_cache,
            "trials": self.historico_trials,
        }
        with open(path_str, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2)
        print(f"[HopfieldKerasTuner] Histórico exportado para: {path_str}")

    def gerar_painel_convergencia(self, path_png: PathType | None = None) -> Figure:
        """Gera painel gráfico ilustrando a evolução dos trials no KerasTuner.

        Parameters
        ----------
        path_png : str | os.PathLike[str] | None, optional
            Caminho opcional para salvar o gráfico como imagem.

        Returns
        -------
        plt.Figure
            Objeto Figure do matplotlib com os eixos formatados.
        """
        if not self.historico_trials:
            raise RuntimeError(
                "Nenhum trial registrado para gerar painel de convergência."
            )

        trials_idx = list(range(1, len(self.historico_trials) + 1))
        f1_vals = [t["metricas"]["val_f1_macro"] for t in self.historico_trials]
        acc_vals = [t["metricas"]["val_acuracia"] for t in self.historico_trials]

        # Melhor F1 acumulado
        melhor_f1_acumulado: list[float] = []
        max_atual = -1.0
        for v in f1_vals:
            if v > max_atual:
                max_atual = v
            melhor_f1_acumulado.append(max_atual)

        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))

        # Painel 1: Convergência F1 Macro
        axs[0].plot(
            trials_idx,
            melhor_f1_acumulado,
            marker="o",
            color="#2ca02c",
            linewidth=2.2,
            label="Melhor F1 Macro (Acumulado)",
        )
        axs[0].scatter(
            trials_idx,
            f1_vals,
            color="#1f77b4",
            alpha=0.6,
            label="F1 Macro por Trial",
        )
        axs[0].set_xlabel("Trial")
        axs[0].set_ylabel("F1 Macro (Validação)")
        axs[0].set_title("Convergência da Otimização KerasTuner")
        axs[0].grid(True, linestyle=":", alpha=0.6)
        axs[0].legend()

        # Painel 2: Acurácia por Trial
        axs[1].plot(
            trials_idx,
            acc_vals,
            marker="s",
            linestyle="--",
            color="#ff7f0e",
            linewidth=1.8,
            label="Acurácia Validação",
        )
        axs[1].set_xlabel("Trial")
        axs[1].set_ylabel("Acurácia Balanceada")
        axs[1].set_title("Evolução da Acurácia por Trial")
        axs[1].grid(True, linestyle=":", alpha=0.6)
        axs[1].legend()

        plt.tight_layout()

        if path_png is not None:
            path_str = str(path_png)
            os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
            fig.savefig(path_str, dpi=300, bbox_inches="tight")
            print(f"[HopfieldKerasTuner] Painel salvo em: {path_str}")

        return fig
