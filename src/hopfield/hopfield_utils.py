"""Funções utilitárias e rotinas matemáticas para redes Hopfield e clustering.

Implementa funções compatíveis com a toolbox legado em MATLAB (sorti, princomp_,
closervects, contaocorr, mat2celllines, wsort, indexa).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


def sorti(
    x: Sequence[Any] | NDArray[Any], mode: Literal["ascend", "descend"] = "ascend"
) -> NDArray[np.intp]:
    """Retorna os índices que ordenam o vetor x (equivalente a sorti.m).

    Parameters
    ----------
    x : Sequence | NDArray
        Vetor numérico de entrada.
    mode : {"ascend", "descend"}, default="ascend"
        Direção da ordenação.

    Returns
    -------
    NDArray[np.intp]
        Vetor de índices ordenados.
    """
    arr: NDArray[Any] = np.asarray(x).ravel()
    idx: NDArray[np.intp] = np.argsort(arr)
    if str(mode).lower().startswith("desc"):
        idx = idx[::-1]
    return idx.copy()


def princomp_(
    W: NDArray[np.float32] | Sequence[Sequence[float]],
) -> NDArray[np.float32]:
    """Calcula PCA sem centralização via decomposição em valores singulares (SVD).

    Equivale a `pca(W, 'Centered', false, 'Algorithm', 'svd')` do MATLAB.

    Parameters
    ----------
    W : NDArray[np.float32] | Sequence
        Matriz de dados (n_amostras × n_features).

    Returns
    -------
    NDArray[np.float32]
        Matriz de loadings (d × d) com autovetores nas colunas ordenados por variância.
    """
    W_arr: NDArray[np.float32] = np.asarray(W, dtype=np.float32)
    _, _, Vt = np.linalg.svd(W_arr, full_matrices=False)
    return Vt.T.astype(np.float32)


def closervects(
    W: NDArray[np.float32] | Sequence[Sequence[float]],
    Wi: NDArray[np.float32] | Sequence[float] | Sequence[int],
    k: int,
    distance: Literal["euclidean"] | float | int | str = "euclidean",
) -> int | NDArray[np.intp]:
    """Identifica os índices dos k vetores em W mais próximos de Wi (equivalente a closervects.m).

    Parameters
    ----------
    W : NDArray[np.float32] | Sequence
        Matriz base contendo os vetores candidatos.
    Wi : NDArray[np.float32] | Sequence
        Vetor de consulta ou índices para cálculo de centróide.
    k : int
        Número de vizinhos mais próximos a retornar.
    distance : {"euclidean"} | float | int | str, default="euclidean"
        Métrica de distância ou norma L-k.

    Returns
    -------
    int | NDArray[np.intp]
        Índice mais próximo (se k=1) ou array com os k índices.
    """
    W_arr: NDArray[np.float32] = np.asarray(W, dtype=np.float32)
    Wi_arr: NDArray[np.float32] = np.asarray(Wi, dtype=np.float32)
    if Wi_arr.ndim == 1:
        Wi_arr = Wi_arr[None, :]
    _, mm = W_arr.shape
    _, q = Wi_arr.shape

    query: NDArray[np.float32]
    if q == mm:
        query = np.asarray(Wi_arr.mean(axis=0), dtype=np.float32)
    else:
        idx: NDArray[np.int_] = Wi_arr.ravel().astype(int) - 1
        query = np.asarray(W_arr[idx].mean(axis=0), dtype=np.float32)

    u: NDArray[np.float32]
    if isinstance(distance, str) and distance.lower() == "euclidean":
        diff = W_arr - query[None, :]
        u = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    else:
        kp: float = float(distance)
        diff_abs = np.abs(W_arr - query[None, :])
        u_sum = np.sum(diff_abs**kp, axis=1)
        if kp >= 1:
            u = (u_sum ** (1.0 / kp)).astype(np.float32)
        else:
            u = u_sum.astype(np.float32)

    ii: NDArray[np.intp] = np.argsort(u)[:k]
    return int(ii[0]) if k == 1 else ii


def contaocorr(v: NDArray[Any] | Sequence[Any], ordby_max: bool = True) -> NDArray[Any]:
    """Conta ocorrências de cada valor distinto em v (equivalente a contaocorr.m).

    Parameters
    ----------
    v : NDArray | Sequence
        Vetor com elementos discretos.
    ordby_max : bool, default=True
        Se True, ordena decrescente por frequência; se False, crescente pelo valor.

    Returns
    -------
    NDArray[Any]
        Matriz (N × 2) onde a coluna 0 é o valor e a coluna 1 é a contagem.
    """
    arr: NDArray[Any] = np.asarray(v).ravel()
    vals, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts, kind="stable") if ordby_max else np.argsort(vals)
    return np.column_stack([vals[order], counts[order]])


def mat2celllines(M: NDArray[Any] | Sequence[Sequence[Any]]) -> list[NDArray[Any]]:
    """Converte as linhas de uma matriz em lista de vetores 1D (equivalente a mat2celllines.m).

    Parameters
    ----------
    M : NDArray | Sequence
        Matriz 2D de entrada.

    Returns
    -------
    list[NDArray]
        Lista onde cada elemento é uma linha da matriz.
    """
    arr: NDArray[Any] = np.asarray(M)
    return [arr[i] for i in range(arr.shape[0])]


def wsort(
    W: NDArray[Any] | sp.spmatrix,
    return_perm: bool = False,
    rng: np.random.Generator | None = None,
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[np.intp]]:
    """Embaralha pseudoaleatoriamente as linhas da matriz W (equivalente a wsort.m).

    Parameters
    ----------
    W : NDArray | sp.spmatrix
        Matriz de dados a ser permutada por linhas.
    return_perm : bool, default=False
        Se True, retorna tupla (matriz_embaralhada, permutacao_indices).
    rng : np.random.Generator | None, optional
        Gerador de números pseudoaleatórios.

    Returns
    -------
    NDArray | Tuple[NDArray, NDArray[np.intp]]
        Matriz com linhas embaralhadas.
    """
    generator: np.random.Generator = rng if rng is not None else np.random.default_rng()
    n_rows: int = int(W.shape[0])
    perm: NDArray[np.intp] = generator.permutation(n_rows)

    out: NDArray[Any]
    if sp.issparse(W):
        out = sp.csr_matrix(W)[perm].toarray().astype(np.float32)
    else:
        out = np.asarray(W)[perm]

    return (out, perm) if return_perm else out


def indexa(
    X: NDArray[Any] | Sequence[Any], xinds: str | int | Sequence[int] | NDArray[np.int_]
) -> Any:
    """Indexação estilo MATLAB 1-based `X(xinds)` (equivalente a indexa.m).

    Parameters
    ----------
    X : NDArray | Sequence
        Array a ser indexado.
    xinds : str | int | Sequence[int] | NDArray
        Índices 1-based ou a string 'SECOND'.

    Returns
    -------
    Any
        Elemento ou subconjunto indexado.
    """
    arr: NDArray[Any] = np.asarray(X)
    if isinstance(xinds, str):
        if xinds.upper() == "SECOND":
            return arr.ravel()[1] if arr.size > 1 else np.array([])
        raise NotImplementedError(f"indexa: modo '{xinds}' não implementado.")
    inds: NDArray[np.int_]
    if isinstance(xinds, (int, np.integer)):
        inds = np.array([int(xinds)], dtype=int) - 1
    else:
        inds = np.asarray(xinds, dtype=int).ravel() - 1
    return arr[inds]
