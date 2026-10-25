"""Módulo de Estratégias de Clusterização para Descoberta de Subpopulações Celulares.

Implementa o padrão Strategy com abordagens de K-Means dinâmico (otimizado via
Calinski-Harabasz), K-Means com k fixo e clustering por densidade com HDBSCAN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


class EstrategiaClusterizacao(ABC):
    """Interface abstrata base para estratégias de clusterização de subconjuntos celulares."""

    @abstractmethod
    def clusterizar(self, Wswp_cls: NDArray[np.float32]) -> NDArray[np.float32]:
        """Recebe os vetores SWeeP da classe atual e calcula os centróides dos subclusters.

        Parameters
        ----------
        Wswp_cls : NDArray[np.float32]
            Matriz de vetores biológicos ou SWeeP da classe (n_células × n_features).

        Returns
        -------
        NDArray[np.float32]
            Matriz de centróides representativos (n_clusters × n_features).
        """
        pass


class EstrategiaKMeansDinamico(EstrategiaClusterizacao):
    """Estratégia que avalia múltiplos valores de k escolhendo o ótimo via Calinski-Harabasz.

    Parameters
    ----------
    k_range : Sequence[int], default=range(2, 16)
        Intervalo de valores de k a testar.
    seed : int, default=42
        Semente para inicialização do K-Means.

    Attributes
    ----------
    k_range : Sequence[int]
        Grade de hiperparâmetros de k.
    seed : int
        Semente pseudoaleatória.
    """

    def __init__(self, k_range: Sequence[int] = range(2, 16), seed: int = 42) -> None:
        self.k_range: Sequence[int] = k_range
        self.seed: int = int(seed)

    def clusterizar(self, Wswp_cls: NDArray[np.float32]) -> NDArray[np.float32]:
        """Executa a busca em grade de k e calcula os centróides ótimos.

        Parameters
        ----------
        Wswp_cls : NDArray[np.float32]
            Matriz de amostras da classe celular.

        Returns
        -------
        NDArray[np.float32]
            Array de centróides calculados.
        """
        n_samples: int = len(Wswp_cls)

        # Se houver menos células do que o k mínimo, retorna a média de todas como centróide
        if n_samples < min(self.k_range):
            print(
                f"[EstrategiaKMeansDinamico] Poucas amostras ({n_samples}). Usando 1 cluster."
            )
            return np.array([Wswp_cls.mean(axis=0)], dtype=np.float32)

        best_score: float = -1.0
        best_k: int = min(self.k_range)
        best_centroids: NDArray[np.float32] | None = None

        # Filtra os k possíveis (não pode ter k >= n_samples)
        valid_ks: list[int] = [k for k in self.k_range if k < n_samples]

        if not valid_ks:
            return np.array([Wswp_cls.mean(axis=0)], dtype=np.float32)

        for k in valid_ks:
            km = KMeans(n_clusters=k, n_init=1, random_state=self.seed)
            labels = km.fit_predict(Wswp_cls)

            if len(set(labels)) < 2:
                continue

            score = float(calinski_harabasz_score(Wswp_cls, labels))
            if score > best_score:
                best_score = score
                best_k = k
                best_centroids = km.cluster_centers_.astype(np.float32)

        if best_centroids is None:
            best_centroids = np.array([Wswp_cls.mean(axis=0)], dtype=np.float32)
            best_k = 1

        print(
            f"[EstrategiaKMeansDinamico] Melhor k encontrado: {best_k} (Calinski-Harabasz: {best_score:.2f})"
        )
        return best_centroids


class EstrategiaKMeansFixo(EstrategiaClusterizacao):
    """Estratégia de particionamento K-Means com número fixo e estático de centróides.

    Parameters
    ----------
    n_clusters : int, default=10
        Número de clusters a extrair.
    seed : int, default=42
        Semente pseudoaleatória.

    Attributes
    ----------
    n_clusters : int
        Contagem fixa de centróides.
    seed : int
        Semente de inicialização.
    """

    def __init__(self, n_clusters: int = 10, seed: int = 42) -> None:
        self.n_clusters: int = int(n_clusters)
        self.seed: int = int(seed)

    def clusterizar(self, Wswp_cls: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calcula os centróides utilizando o valor fixo `n_clusters`.

        Parameters
        ----------
        Wswp_cls : NDArray[np.float32]
            Matriz de amostras da classe.

        Returns
        -------
        NDArray[np.float32]
            Centróides calculados.
        """
        n_samples: int = len(Wswp_cls)

        if n_samples <= self.n_clusters:
            print(
                f"[EstrategiaKMeansFixo] Células insuficientes ({n_samples}) para {self.n_clusters} clusters. Retornando todas como centróides."
            )
            return np.asarray(Wswp_cls, dtype=np.float32)

        km = KMeans(n_clusters=self.n_clusters, n_init=1, random_state=self.seed)
        km.fit(Wswp_cls)
        return km.cluster_centers_.astype(np.float32)


class EstrategiaHDBSCAN(EstrategiaClusterizacao):
    """Estratégia de clusterização hierárquica baseada em densidade com descarte de ruído.

    Parameters
    ----------
    min_cluster_size : int, default=15
        Tamanho mínimo de vizinhança para formação de cluster.

    Attributes
    ----------
    min_cluster_size : int
        Hiperparâmetro de densidade mínima.
    """

    def __init__(self, min_cluster_size: int = 15) -> None:
        self.min_cluster_size: int = int(min_cluster_size)

    def clusterizar(self, Wswp_cls: NDArray[np.float32]) -> NDArray[np.float32]:
        """Executa HDBSCAN e extrai o centróide médio de cada cluster denso identificado.

        Parameters
        ----------
        Wswp_cls : NDArray[np.float32]
            Matriz de amostras da classe.

        Returns
        -------
        NDArray[np.float32]
            Array de centróides densos.
        """
        try:
            import hdbscan  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "O pacote 'hdbscan' não está instalado. Rode 'pip install hdbscan' no seu terminal."
            ) from err

        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        labels = clusterer.fit_predict(Wswp_cls)

        unique_labels = set(labels)
        centroides: list[NDArray[np.float32]] = []

        for c_id in unique_labels:
            if c_id == -1:  # Ignora o ruído
                continue

            cluster_points = Wswp_cls[labels == c_id]
            centroid = cluster_points.mean(axis=0).astype(np.float32)
            centroides.append(centroid)

        if not centroides:
            print(
                "[EstrategiaHDBSCAN] Nenhum cluster denso encontrado. Usando a média global."
            )
            return np.array([Wswp_cls.mean(axis=0)], dtype=np.float32)

        centroides_arr: NDArray[np.float32] = np.vstack(centroides).astype(np.float32)
        print(
            f"[EstrategiaHDBSCAN] Encontrados {len(centroides)} clusters densos válidos."
        )
        return centroides_arr
