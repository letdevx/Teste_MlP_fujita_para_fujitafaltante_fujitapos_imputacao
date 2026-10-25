"""Módulo de Redes de Memória Associativa Moderna (Modern Hopfield Networks).

Implementa a dinâmica de atenção contínua de Ramsauer et al. (2020) com
capacidade de armazenamento exponencial para recuperação e imputação scRNA-seq.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

PathType = str | os.PathLike[str]
InputQueries = NDArray[Any] | sp.spmatrix | torch.Tensor


class ModernHopfieldNetwork(nn.Module):
    """Rede de Hopfield Moderna / Dense Associative Memory (Ramsauer et al., 2020).

    Substitui hopf_tr (treino) e hopf_ts (teste) do script MATLAB original.
    Capacidade de armazenamento exponencial em vez de linear; recuperação
    equivalente a um passo de atenção softmax: softmax(β·Ξ·ξ)·Ξᵀ.

    Parameters
    ----------
    beta : float, default=8.0
        Temperatura inversa do softmax (maior → comportamento winner-takes-all).
    n_iters : int, default=1
        Número de iterações síncronas da regra de atualização.
    binary : bool, default=True
        Se True, mapeia os padrões para o espaço bipolar {-1, +1} e binariza a saída em {0, 1}.
    threshold : float, default=0.0
        Limiar de corte para binarizar a saída quando `binary=True`.
    normalize : bool, default=False
        Se True, aplica normalização L2 (similaridade cosseno esférica) nas queries e padrões.
    scale_by_dim : bool, default=False
        Se True e `normalize=False`, divide os produtos escalares por sqrt(D),
        estabilizando a variância dos logits para O(1) e evitando saturação de atenção.

    Attributes
    ----------
    beta : float
        Parâmetro de temperatura inversa.
    n_iters : int
        Número de iterações.
    binary : bool
        Indicador de dinâmica bipolar/binária.
    threshold : float
        Limiar de ativação.
    normalize : bool
        Indicador de normalização esférica.
    scale_by_dim : bool
        Indicador de escalonamento dimensional por raiz quadrada de D.
    patterns : torch.Tensor
        Tensor com os padrões biológicos armazenados em memória.
    """

    patterns: torch.Tensor

    def __init__(
        self,
        beta: float = 8.0,
        n_iters: int = 1,
        binary: bool = True,
        threshold: float = 0.0,
        normalize: bool = False,
        scale_by_dim: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.beta: float = float(beta)
        self.n_iters: int = int(n_iters)
        self.binary: bool = bool(binary)
        self.threshold: float = float(threshold)
        self.normalize: bool = bool(normalize)
        self.scale_by_dim: bool = bool(scale_by_dim)
        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.register_buffer("patterns", torch.empty(0, dtype=torch.float32))

    def store(
        self, patterns: NDArray[Any] | torch.Tensor | Sequence[Sequence[float]]
    ) -> ModernHopfieldNetwork:
        """Armazena os padrões na matriz de memória da rede.

        Parameters
        ----------
        patterns : NDArray | torch.Tensor | Sequence
            Matriz de padrões (n_patterns × n_features) a serem memorizados.

        Returns
        -------
        ModernHopfieldNetwork
            A própria instância para encadeamento.
        """
        arr: NDArray[np.float32] = np.asarray(patterns, dtype=np.float32)
        K: torch.Tensor = torch.as_tensor(arr, dtype=torch.float32)

        if self.binary:
            K = 2.0 * K - 1.0

        self.patterns = K.to(self.device)
        print(
            f"[ModernHopfieldNetwork] {self.patterns.shape[0]} padrões armazenados "
            f"({self.patterns.shape[1]} genes, device={self.device.type})"
        )
        return self

    @overload
    def retrieve(
        self,
        queries: InputQueries,
        batch_size: int = 1024,
        normalize: bool | None = None,
        subspace_mask: NDArray[Any] | torch.Tensor | None = None,
        mask_sentinela_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
        fill_value: float = 0.5,
        out_buffer: NDArray[np.float32] | None = None,
        *,
        return_probabilities: Literal[True],
        return_attention_weights: Literal[True],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...

    @overload
    def retrieve(
        self,
        queries: InputQueries,
        batch_size: int = 1024,
        normalize: bool | None = None,
        subspace_mask: NDArray[Any] | torch.Tensor | None = None,
        mask_sentinela_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
        fill_value: float = 0.5,
        out_buffer: NDArray[np.float32] | None = None,
        *,
        return_probabilities: Literal[True],
        return_attention_weights: Literal[False] = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...

    @overload
    def retrieve(
        self,
        queries: InputQueries,
        batch_size: int = 1024,
        normalize: bool | None = None,
        subspace_mask: NDArray[Any] | torch.Tensor | None = None,
        mask_sentinela_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
        fill_value: float = 0.5,
        out_buffer: NDArray[np.float32] | None = None,
        *,
        return_probabilities: Literal[False] = False,
        return_attention_weights: Literal[True],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...

    @overload
    def retrieve(
        self,
        queries: InputQueries,
        batch_size: int = 1024,
        normalize: bool | None = None,
        subspace_mask: NDArray[Any] | torch.Tensor | None = None,
        mask_sentinela_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
        fill_value: float = 0.5,
        out_buffer: NDArray[np.float32] | None = None,
        return_probabilities: Literal[False] = False,
        return_attention_weights: Literal[False] = False,
    ) -> NDArray[np.float32]: ...

    @torch.no_grad()
    def retrieve(
        self,
        queries: InputQueries,
        batch_size: int = 1024,
        normalize: bool | None = None,
        subspace_mask: NDArray[Any] | torch.Tensor | None = None,
        mask_sentinela_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
        fill_value: float = 0.5,
        out_buffer: NDArray[np.float32] | None = None,
        return_probabilities: bool = False,
        return_attention_weights: bool = False,
    ) -> Any:
        """Recupera o padrão de memória associativa mais próximo para cada query.

        Parameters
        ----------
        queries : InputQueries
            Matriz de entrada (esparsa CSR ou densa numpy).
        batch_size : int, default=1024
            Tamanho do lote de processamento OOM-Safe em memória.
        normalize : bool | None, optional
            Se fornecido, sobrescreve o comportamento de normalização L2 da instância.
        subspace_mask : NDArray | torch.Tensor | None, optional
            Máscara de colunas para cálculo da atenção em subespaço restrito.
        mask_sentinela_ausentes : NDArray[bool] | Sequence[int] | None, optional
            Máscara booleana ou lista de índices de genes ausentes para injeção de `fill_value`.
        fill_value : float, default=0.5
            Valor atribuído aos genes ausentes (padrão 0.5 -> 0.0 no espaço bipolar).
        out_buffer : NDArray[np.float32] | None, optional
            Buffer pré-alocado opcional para escrita direta dos resultados.
        return_probabilities : bool, default=False
            Se True, retorna uma tupla com probabilidades contínuas [0, 1].
        return_attention_weights : bool, default=False
            Se True, retorna também a matriz de pesos de atenção Softmax da Hopfield
            (n_queries × n_padroes), habilitando Softmax Class Pooling.

        Returns
        -------
        NDArray[np.float32] | tuple
            Matriz recuperada e imputada (ou tupla contendo probabilidades contínuas
            e/ou pesos de atenção Softmax).
        """
        if self.patterns.numel() == 0:
            raise RuntimeError(
                "[ModernHopfieldNetwork] Execute .store() antes de .retrieve()."
            )

        norm_active: bool = (
            normalize if normalize is not None else getattr(self, "normalize", False)
        )

        Xi: torch.Tensor = self.patterns.to(dtype=torch.float32, device=self.device)
        n_patterns: int = int(Xi.shape[0])
        is_sparse: bool = sp.issparse(queries)

        n_queries: int
        n_features: int
        queries_np: NDArray[Any] | None

        queries_csr: sp.csr_matrix | None = None
        if is_sparse:
            queries_csr = sp.csr_matrix(queries)
            assert queries_csr is not None and queries_csr.shape is not None
            n_queries = int(queries_csr.shape[0])
            n_features = int(queries_csr.shape[1])
            queries_np = None
        else:
            if isinstance(queries, torch.Tensor):
                queries_np = queries.detach().cpu().numpy()
            else:
                queries_np = np.asarray(queries)
            assert queries_np is not None and queries_np.shape is not None
            n_queries = int(queries_np.shape[0])
            n_features = int(queries_np.shape[1])

        subspace_tensor: torch.Tensor | None = None
        if subspace_mask is not None:
            if isinstance(subspace_mask, np.ndarray):
                subspace_tensor = torch.from_numpy(subspace_mask).to(device=self.device)
            elif isinstance(subspace_mask, torch.Tensor):
                subspace_tensor = subspace_mask.to(device=self.device)

        if out_buffer is None:
            out_buffer = np.empty((n_queries, n_features), dtype=np.float32)

        prob_buffer: NDArray[np.float32] | None = None
        if return_probabilities:
            prob_buffer = np.empty((n_queries, n_features), dtype=np.float32)

        att_buffer: NDArray[np.float32] | None = None
        if return_attention_weights:
            att_buffer = np.empty((n_queries, n_patterns), dtype=np.float32)

        Xi_att: torch.Tensor = (
            Xi[:, subspace_tensor] if subspace_tensor is not None else Xi
        )
        Xi_norm: torch.Tensor | None = None
        if norm_active:
            Xi_norm = F.normalize(Xi_att, p=2, dim=-1, eps=1e-8)

        for s in range(0, n_queries, batch_size):
            chunk_np: NDArray[np.float32]
            if is_sparse:
                assert queries_csr is not None
                chunk_np = queries_csr[s : s + batch_size].toarray().astype(np.float32)
            else:
                assert queries_np is not None
                chunk_np = queries_np[s : s + batch_size].astype(
                    np.float32,
                    copy=mask_sentinela_ausentes is not None,
                )

            if mask_sentinela_ausentes is not None:
                chunk_np[:, mask_sentinela_ausentes] = fill_value

            x: torch.Tensor = torch.from_numpy(chunk_np).to(device=self.device)

            if self.binary:
                x = 2.0 * x - 1.0

            weights: torch.Tensor | None = None
            for _ in range(self.n_iters):
                x_att: torch.Tensor = (
                    x[:, subspace_tensor] if subspace_tensor is not None else x
                )

                if norm_active:
                    assert Xi_norm is not None
                    x_norm = F.normalize(x_att, p=2, dim=-1, eps=1e-8)
                    scores = self.beta * (x_norm @ Xi_norm.T)
                else:
                    escala = (
                        (1.0 / np.sqrt(float(x_att.shape[-1])))
                        if getattr(self, "scale_by_dim", False)
                        else 1.0
                    )
                    scores = (self.beta * escala) * (x_att @ Xi_att.T)
                weights = torch.softmax(scores, dim=-1)
                x = weights @ Xi

            if att_buffer is not None:
                assert weights is not None
                att_buffer[s : s + batch_size] = (
                    weights.detach().cpu().numpy().astype(np.float32)
                )

            if prob_buffer is not None:
                if self.binary:
                    p_tensor = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
                else:
                    p_tensor = torch.clamp(x, 0.0, 1.0)
                prob_buffer[s : s + batch_size] = (
                    p_tensor.detach().cpu().numpy().astype(np.float32)
                )

            if self.binary:
                x = (x > self.threshold).float()

            res_np: NDArray[np.float32] = x.detach().cpu().numpy().astype(np.float32)
            out_buffer[s : s + batch_size] = res_np
            del x, chunk_np

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"[ModernHopfieldNetwork] Recuperação concluída: {out_buffer.shape} (dtype={out_buffer.dtype})"
        )
        if return_probabilities and return_attention_weights:
            assert prob_buffer is not None and att_buffer is not None
            return out_buffer, prob_buffer, att_buffer
        if return_probabilities:
            assert prob_buffer is not None
            return out_buffer, prob_buffer
        if return_attention_weights:
            assert att_buffer is not None
            return out_buffer, att_buffer
        return out_buffer

    @torch.no_grad()
    def compute_attention_weights(
        self,
        queries: InputQueries,
        batch_size: int = 2048,
        normalize: bool | None = None,
        subspace_mask: NDArray[Any] | torch.Tensor | None = None,
        mask_sentinela_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
        fill_value: float = 0.5,
    ) -> NDArray[np.float32]:
        """Calcula os pesos de atenção Softmax da Hopfield sobre os padrões armazenados.

        Evita reconstruir o perfil completo de genes quando apenas a atenção / classificação
        por Softmax Class Pooling for necessária, economizando memória e tempo de execução.

        Parameters
        ----------
        queries : InputQueries
            Matriz de entrada (esparsa CSR ou densa).
        batch_size : int, default=2048
            Tamanho do lote OOM-Safe.
        normalize : bool | None, optional
            Sobrescreve a normalização L2 da rede.
        subspace_mask : NDArray | torch.Tensor | None, optional
            Máscara de subespaço para cálculo da atenção.
        mask_sentinela_ausentes : NDArray[bool] | Sequence[int] | None, optional
            Máscara de genes ausentes para injeção de fill_value.
        fill_value : float, default=0.5
            Valor dos genes ausentes (0.5 vira 0.0 no espaço bipolar).

        Returns
        -------
        NDArray[np.float32]
            Matriz de pesos de atenção Softmax (n_queries × n_padroes).
        """
        if self.patterns.numel() == 0:
            raise RuntimeError(
                "[ModernHopfieldNetwork] Execute .store() antes de .compute_attention_weights()."
            )

        norm_active: bool = (
            normalize if normalize is not None else getattr(self, "normalize", False)
        )

        Xi: torch.Tensor = self.patterns.to(dtype=torch.float32, device=self.device)
        n_patterns: int = int(Xi.shape[0])
        is_sparse: bool = sp.issparse(queries)

        n_queries: int
        queries_np: NDArray[Any] | None
        queries_csr: sp.csr_matrix | None = None

        if is_sparse:
            queries_csr = sp.csr_matrix(queries)
            assert queries_csr is not None and queries_csr.shape is not None
            n_queries = int(queries_csr.shape[0])
            queries_np = None
        else:
            if isinstance(queries, torch.Tensor):
                queries_np = queries.detach().cpu().numpy()
            else:
                queries_np = np.asarray(queries)
            assert queries_np is not None and queries_np.shape is not None
            n_queries = int(queries_np.shape[0])

        subspace_tensor: torch.Tensor | None = None
        if subspace_mask is not None:
            if isinstance(subspace_mask, np.ndarray):
                subspace_tensor = torch.from_numpy(subspace_mask).to(device=self.device)
            elif isinstance(subspace_mask, torch.Tensor):
                subspace_tensor = subspace_mask.to(device=self.device)

        Xi_att: torch.Tensor = (
            Xi[:, subspace_tensor] if subspace_tensor is not None else Xi
        )
        Xi_norm: torch.Tensor | None = None
        if norm_active:
            Xi_norm = F.normalize(Xi_att, p=2, dim=-1, eps=1e-8)

        att_buffer: NDArray[np.float32] = np.empty(
            (n_queries, n_patterns), dtype=np.float32
        )

        for s in range(0, n_queries, batch_size):
            chunk_np: NDArray[np.float32]
            if is_sparse:
                assert queries_csr is not None
                chunk_np = queries_csr[s : s + batch_size].toarray().astype(np.float32)
            else:
                assert queries_np is not None
                chunk_np = queries_np[s : s + batch_size].astype(
                    np.float32,
                    copy=mask_sentinela_ausentes is not None,
                )

            if mask_sentinela_ausentes is not None:
                chunk_np[:, mask_sentinela_ausentes] = fill_value

            x: torch.Tensor = torch.from_numpy(chunk_np).to(device=self.device)
            if self.binary:
                x = 2.0 * x - 1.0

            weights: torch.Tensor | None = None
            for _ in range(self.n_iters):
                x_att: torch.Tensor = (
                    x[:, subspace_tensor] if subspace_tensor is not None else x
                )
                if norm_active:
                    assert Xi_norm is not None
                    x_norm = F.normalize(x_att, p=2, dim=-1, eps=1e-8)
                    scores = self.beta * (x_norm @ Xi_norm.T)
                else:
                    escala = (
                        (1.0 / np.sqrt(float(x_att.shape[-1])))
                        if getattr(self, "scale_by_dim", False)
                        else 1.0
                    )
                    scores = (self.beta * escala) * (x_att @ Xi_att.T)
                weights = torch.softmax(scores, dim=-1)
                if self.n_iters > 1:
                    x = weights @ Xi

            assert weights is not None
            att_buffer[s : s + batch_size] = weights.detach().cpu().numpy().astype(np.float32)
            del x, chunk_np

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return att_buffer

    def salvar(self, path: PathType) -> ModernHopfieldNetwork:
        """Salva os parâmetros da rede e a matriz de padrões em disco (.pt).

        Parameters
        ----------
        path : str | os.PathLike[str]
            Caminho do arquivo checkpoint PyTorch.

        Returns
        -------
        ModernHopfieldNetwork
            A própria instância.
        """
        if self.patterns.numel() == 0:
            raise RuntimeError(
                "[ModernHopfieldNetwork] Execute .store() antes de salvar."
            )
        path_str: str = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
        torch.save(
            {
                "beta": self.beta,
                "n_iters": self.n_iters,
                "binary": self.binary,
                "threshold": self.threshold,
                "normalize": getattr(self, "normalize", False),
                "scale_by_dim": getattr(self, "scale_by_dim", False),
                "patterns": self.patterns.detach().cpu(),
            },
            path_str,
        )
        print(
            f"[ModernHopfieldNetwork] Rede salva em: {path_str} "
            f"({self.patterns.shape[0]} padrões)"
        )
        return self

    @classmethod
    def carregar(
        cls, path: PathType, device: str | torch.device | None = None
    ) -> ModernHopfieldNetwork:
        """Carrega uma rede previamente treinada a partir de um checkpoint PyTorch (.pt).

        Parameters
        ----------
        path : str | os.PathLike[str]
            Caminho do arquivo .pt.
        device : str | torch.device | None, optional
            Dispositivo de execução ('cuda' ou 'cpu'). Se None, auto-detecta.

        Returns
        -------
        ModernHopfieldNetwork
            Instância carregada com parâmetros e tensores restaurados.
        """
        path_str: str = str(path)
        data = torch.load(path_str, map_location="cpu")
        rede = cls(
            beta=data["beta"],
            n_iters=data["n_iters"],
            binary=data["binary"],
            threshold=data["threshold"],
            normalize=data.get("normalize", False),
            scale_by_dim=data.get("scale_by_dim", False),
            device=device,
        )
        rede.patterns = data["patterns"].to(rede.device)
        print(
            f"[ModernHopfieldNetwork] Rede carregada de: {path_str} "
            f"({rede.patterns.shape[0]} padrões, device={rede.device.type})"
        )
        return rede

    def salvar_com_metadados(
        self,
        path_pt: PathType,
        path_meta: PathType,
        meta: Sequence[Any],
        classes: Sequence[int] | None = None,
        nc: int | None = None,
    ) -> ModernHopfieldNetwork:
        """Salva a rede (.pt) e o arquivo JSON de metadados (.json).

        Parameters
        ----------
        path_pt : str | os.PathLike[str]
            Caminho do checkpoint binário .pt.
        path_meta : str | os.PathLike[str]
            Caminho do arquivo JSON de metadados estruturais.
        meta : Sequence[Any]
            Lista com identificadores (classe, cluster) de cada padrão.
        classes : Sequence[int] | None, optional
            Rótulos das classes biológicas.
        nc : int | None, optional
            Número de centróides por classe.

        Returns
        -------
        ModernHopfieldNetwork
            A própria instância.
        """
        self.salvar(path_pt)

        meta_serializavel: list[Any] = [
            list(item) if isinstance(item, (tuple, list, np.ndarray)) else item
            for item in meta
        ]
        n_patterns: int = int(self.patterns.shape[0]) if self.patterns.numel() else 0
        n_genes: int = int(self.patterns.shape[1]) if self.patterns.numel() else 0
        cls_list: list[int] = (
            list(classes) if classes is not None else [1, 2, 3, 4, 5, 6, 7]
        )
        nc_val: int = (
            nc if nc is not None else (n_patterns // len(cls_list) if cls_list else 30)
        )

        info: dict[str, Any] = {
            "meta": meta_serializavel,
            "classes": cls_list,
            "nc": nc_val,
            "n_patterns": n_patterns,
            "n_genes": n_genes,
        }

        path_meta_str: str = str(path_meta)
        os.makedirs(os.path.dirname(os.path.abspath(path_meta_str)), exist_ok=True)
        with open(path_meta_str, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        print(
            f"[ModernHopfieldNetwork] Metadados salvos em: {path_meta_str} "
            f"({n_patterns} padrões, {n_genes} genes)"
        )
        return self

    @classmethod
    def carregar_com_metadados(
        cls,
        path_pt: PathType,
        path_meta: PathType,
        device: str | torch.device | None = None,
    ) -> tuple[ModernHopfieldNetwork, list[tuple[int, int]], dict[str, Any]]:
        """Carrega a rede (.pt) e o arquivo JSON de metadados (.json).

        Parameters
        ----------
        path_pt : str | os.PathLike[str]
            Caminho do arquivo .pt.
        path_meta : str | os.PathLike[str]
            Caminho do arquivo .json.
        device : str | torch.device | None, optional
            Dispositivo de execução ('cuda' ou 'cpu'). Se None, auto-detecta.

        Returns
        -------
        Tuple[ModernHopfieldNetwork, list[tuple[int, int]], dict[str, Any]]
            Tupla contendo a rede restaurada, a lista de metadados e o dicionário JSON bruto.
        """
        rede = cls.carregar(path_pt, device=device)

        path_meta_str: str = str(path_meta)
        with open(path_meta_str, encoding="utf-8") as f:
            meta_json: dict[str, Any] = json.load(f)

        meta_eval: list[tuple[int, int]] = [tuple(x) for x in meta_json["meta"]]  # type: ignore[misc]

        print(
            f"[ModernHopfieldNetwork] Metadados carregados de: {path_meta_str} "
            f"(classes={meta_json.get('classes')}, nc={meta_json.get('nc')}, n_patterns={meta_json.get('n_patterns')})"
        )
        return rede, meta_eval, meta_json

    def hopf_tr(
        self, patterns: NDArray[Any] | torch.Tensor | Sequence[Sequence[float]]
    ) -> ModernHopfieldNetwork:
        """Alias compatível com o script MATLAB original para treino."""
        return self.store(patterns)

    def hopf_ts(self, queries: InputQueries, **kw: Any) -> NDArray[np.float32]:
        """Alias compatível com o script MATLAB original para teste."""
        return self.retrieve(queries, **kw)

    def forward(self, queries: InputQueries, **kw: Any) -> NDArray[np.float32]:
        """Encaminhamento de forward padrão PyTorch delegando para retrieve."""
        return self.retrieve(queries, **kw)

    def __repr__(self) -> str:
        """Representação textual da rede Hopfield moderna."""
        n_pad: int = int(self.patterns.shape[0]) if self.patterns.numel() else 0
        dim: int = int(self.patterns.shape[1]) if self.patterns.numel() else 0
        return (
            f"ModernHopfieldNetwork(\n"
            f"  beta       = {self.beta}\n"
            f"  n_iters    = {self.n_iters}\n"
            f"  binary     = {self.binary}\n"
            f"  threshold  = {self.threshold}\n"
            f"  normalize  = {getattr(self, 'normalize', False)}\n"
            f"  scale_by_d = {getattr(self, 'scale_by_dim', False)}\n"
            f"  patterns   = {n_pad} × {dim}\n"
            f")"
        )
