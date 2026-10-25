"""Módulo de Avaliação de Recuperação e Classificação de Memória Hopfield.

Compara matrizes imputadas pela rede Hopfield com padrões de referência
e rótulos verdadeiros de tipos celulares / estágios patológicos.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class AvaliadorHopfield:
    """Avalia os resultados da rede Hopfield com métricas de classificação e recuperação.

    Compara os padrões recuperados pela rede com os rótulos verdadeiros,
    mapeando cada padrão recuperado para a classe do protótipo mais próximo em `padroes`.

    Parameters
    ----------
    padroes : NDArray | Sequence
        Padrões de protótipos armazenados na rede.
    classes : Sequence[int]
        Lista ordenada de classes de interesse.
    nc : int, default=10
        Número de subclusters ou protótipos por classe.
    nomes_classes : Sequence[str] | None, optional
        Nomes legíveis das classes para exibição gráfica.
    meta : Sequence[tuple[int, int]] | None, optional
        Metadados `(classe, idx)` de cada protótipo.
    metrica : str, default="euclidiana"
        Métrica de distância para projeção ("euclidiana" ou "cosseno").

    Attributes
    ----------
    padroes : NDArray[np.float32]
        Matriz de protótipos biológicos.
    classes : list[int]
        Rótulos das classes analisadas.
    nc : int
        Centróides por classe.
    nomes_classes : list[str] | None
        Rótulos textuais.
    metrica : str
        Métrica ativa.
    acuracia : float | None
        Acurácia global balanceada.
    f1_macro : float | None
        Score F1 macro.
    f1_weighted : float | None
        Score F1 ponderado.
    taxa_reconstrucao : float | None
        Taxa de recuperação exata (Hamming zero).
    semelhanca_media : float | None
        Similaridade média (1 - Hamming normalizado).
    matriz_conf : NDArray[np.int_] | None
        Matriz de confusão.
    y_true : NDArray[np.int_] | None
        Vetor de classes reais.
    y_pred : NDArray[np.int_] | None
        Vetor de classes preditas.
    idx_proto : NDArray[np.intp] | None
        Índices dos protótipos associados.
    """

    def __init__(
        self,
        padroes: NDArray[Any] | Sequence[Sequence[float]],
        classes: Sequence[int],
        nc: int = 10,
        nomes_classes: Sequence[str] | None = None,
        meta: Sequence[tuple[int, int]] | None = None,
        metrica: str = "euclidiana",
    ) -> None:
        self.padroes: NDArray[np.float32] = np.asarray(padroes, dtype=np.float32)
        self.classes: list[int] = list(classes)
        self.nc: int = int(nc)
        self.nomes_classes: list[str] | None = (
            list(nomes_classes) if nomes_classes is not None else None
        )
        self.metrica: str = str(metrica).lower()
        self._pattern_classes: NDArray[np.int_] | None = (
            np.array([m[0] for m in meta], dtype=int) if meta is not None else None
        )
        self.acuracia: float | None = None
        self.f1_macro: float | None = None
        self.f1_weighted: float | None = None
        self.taxa_reconstrucao: float | None = None
        self.semelhanca_media: float | None = None
        self.matriz_conf: NDArray[np.int_] | None = None
        self.y_true: NDArray[np.int_] | None = None
        self.y_pred: NDArray[np.int_] | None = None
        self.idx_proto: NDArray[np.intp] | None = None
        self.prob_classes: NDArray[np.float32] | None = None

    def avaliar_por_atencao(
        self,
        attention_weights: NDArray[Any] | Sequence[Sequence[float]],
        labels: NDArray[Any] | Sequence[int],
        priors: NDArray[Any] | Sequence[float] | None = None,
    ) -> AvaliadorHopfield:
        """Avalia a classificação via Softmax Class Pooling (Atenção Agregada da Hopfield).

        Agrega a probabilidade de atenção de todos os subclusters pertencentes a cada
        classe canônica para formar a distribuição a posteriori P(classe | célula),
        eliminando a fragilidade e o viés do 1-NN pontual.

        Parameters
        ----------
        attention_weights : NDArray | Sequence
            Matriz de pesos de atenção Softmax da Hopfield (n_células × n_padrões).
        labels : NDArray | Sequence
            Vetor com os rótulos verdadeiros de cada célula.
        priors : NDArray | Sequence, optional
            Pesos a priori por classe para balanceamento opcional.

        Returns
        -------
        AvaliadorHopfield
            A própria instância com as métricas preenchidas.
        """
        classes_arr: NDArray[np.int_] = np.array(self.classes, dtype=int)
        labels_arr: NDArray[np.int_] = np.asarray(labels, dtype=int)
        att_arr: NDArray[np.float32] = np.asarray(attention_weights, dtype=np.float32)

        n_obs: int = int(att_arr.shape[0])
        n_padroes: int = int(att_arr.shape[1])

        pattern_classes: NDArray[np.int_]
        if self._pattern_classes is not None:
            pattern_classes = self._pattern_classes
        else:
            pattern_classes = np.array(
                [classes_arr[j // self.nc] for j in range(n_padroes)], dtype=int
            )

        prob_matrix: NDArray[np.float32] = np.zeros(
            (n_obs, len(self.classes)), dtype=np.float32
        )
        for k, c in enumerate(self.classes):
            cols = np.where(pattern_classes == c)[0]
            if len(cols) > 0:
                prob_matrix[:, k] = att_arr[:, cols].sum(axis=1)

        if priors is not None:
            priors_arr: NDArray[np.float32] = np.asarray(priors, dtype=np.float32)
            prob_matrix = prob_matrix * priors_arr
            s = prob_matrix.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            prob_matrix = (prob_matrix / s).astype(np.float32)

        self.prob_classes = prob_matrix
        pred: NDArray[np.int_] = classes_arr[prob_matrix.argmax(axis=1)]
        self.idx_proto = np.asarray(att_arr.argmax(axis=1), dtype=np.intp)

        mask: NDArray[np.bool_] = np.isin(labels_arr, self.classes)
        self.y_true = labels_arr[mask]
        self.y_pred = pred[mask]

        self.acuracia = float((self.y_true == self.y_pred).mean())
        self.f1_macro = float(
            f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        )
        self.f1_weighted = float(
            f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        )
        self.matriz_conf = confusion_matrix(
            self.y_true, self.y_pred, labels=self.classes
        )

        print(
            f"[AvaliadorHopfield - Softmax Class Pooling] Acurácia: {self.acuracia * 100:.2f}% (n={mask.sum():,})"
        )
        print(
            f"[AvaliadorHopfield - Softmax Class Pooling] F1 macro={self.f1_macro:.4f}, F1 ponderado={self.f1_weighted:.4f}"
        )
        return self

    def avaliar(
        self,
        Wrecuperado: NDArray[Any] | Sequence[Sequence[float]],
        labels: NDArray[Any] | Sequence[int],
    ) -> AvaliadorHopfield:
        """Avalia a recuperação comparando com os rótulos verdadeiros.

        Parameters
        ----------
        Wrecuperado : NDArray | Sequence
            Matriz de estados finais recuperados pela rede Hopfield.
        labels : NDArray | Sequence
            Vetor com os rótulos verdadeiros correspondentes a cada linha.

        Returns
        -------
        AvaliadorHopfield
            A própria instância com as métricas preenchidas.
        """
        classes_arr: NDArray[np.int_] = np.array(self.classes, dtype=int)
        labels_arr: NDArray[np.int_] = np.asarray(labels, dtype=int)
        W_arr: NDArray[np.float32] = np.asarray(Wrecuperado, dtype=np.float32)

        print(
            f"[AvaliadorHopfield] Mapeando padrões recuperados para classes (métrica: {self.metrica}, lotes em float32)...",
            flush=True,
        )
        n_obs: int = int(W_arr.shape[0])
        n_genes: int = int(W_arr.shape[1])
        perf_f: NDArray[np.float32] = self.padroes.astype(np.float32, copy=False)
        b2: NDArray[np.float32] = (perf_f**2).sum(axis=1, keepdims=True).T
        perf_norms: NDArray[np.float32] = np.linalg.norm(perf_f, axis=1, keepdims=True)
        perf_norms[perf_norms == 0] = 1.0
        perf_f_norm: NDArray[np.float32] = perf_f / perf_norms

        idx_proto_list: list[NDArray[np.intp]] = []
        hamming_list: list[NDArray[np.float32]] = []
        chunk_size: int = 4096

        for start in range(0, n_obs, chunk_size):
            end: int = min(start + chunk_size, n_obs)
            W_chunk_f: NDArray[np.float32] = np.asarray(
                W_arr[start:end], dtype=np.float32
            )

            idx_chunk: NDArray[np.intp]
            min_sq_dist: NDArray[np.float32]

            if self.metrica == "cosseno":
                w_norms = np.linalg.norm(W_chunk_f, axis=1, keepdims=True)
                w_norms[w_norms == 0] = 1.0
                w_norm = W_chunk_f / w_norms
                sim_matrix = w_norm @ perf_f_norm.T
                idx_chunk = np.asarray(sim_matrix.argmax(axis=1), dtype=np.intp)
                diff = W_chunk_f - perf_f[idx_chunk]
                min_sq_dist = (diff**2).sum(axis=1)
            else:
                a2_chunk = (W_chunk_f**2).sum(axis=1, keepdims=True)
                sq_dist_chunk = a2_chunk + b2 - 2 * (W_chunk_f @ perf_f.T)
                idx_chunk = np.asarray(sq_dist_chunk.argmin(axis=1), dtype=np.intp)
                min_sq_dist = sq_dist_chunk[np.arange(end - start), idx_chunk]

            hamming_chunk = (np.maximum(0.0, min_sq_dist) / n_genes).astype(np.float32)

            idx_proto_list.append(idx_chunk)
            hamming_list.append(hamming_chunk)

        idx_proto: NDArray[np.intp] = np.concatenate(idx_proto_list)
        hamming: NDArray[np.float32] = np.concatenate(hamming_list).astype(np.float32)
        self.idx_proto = idx_proto

        pred: NDArray[np.int_]
        if self._pattern_classes is not None:
            pred = self._pattern_classes[idx_proto]
        else:
            pred = classes_arr[idx_proto // self.nc]

        mask: NDArray[np.bool_] = np.isin(labels_arr, self.classes)
        self.y_true = labels_arr[mask]
        self.y_pred = pred[mask]

        self.acuracia = float((self.y_true == self.y_pred).mean())
        self.f1_macro = float(
            f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        )
        self.f1_weighted = float(
            f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        )
        self.taxa_reconstrucao = float((hamming[mask] == 0).mean())
        self.semelhanca_media = float((1 - hamming[mask]).mean())
        self.matriz_conf = confusion_matrix(
            self.y_true, self.y_pred, labels=self.classes
        )

        print(
            f"[AvaliadorHopfield] Acurácia: {self.acuracia * 100:.2f}% (n={mask.sum():,})"
        )
        print(
            f"[AvaliadorHopfield] F1 macro={self.f1_macro:.4f}, F1 ponderado={self.f1_weighted:.4f}"
        )
        print(
            f"[AvaliadorHopfield] Taxa de reconstrução exata : {self.taxa_reconstrucao * 100:.2f}%"
        )
        print(
            f"[AvaliadorHopfield] Semelhança média ao protótipo: {self.semelhanca_media:.4f}"
        )
        rotulos_rep: list[str] = (
            self.nomes_classes if self.nomes_classes else [str(c) for c in self.classes]
        )
        print(
            classification_report(
                self.y_true,
                self.y_pred,
                labels=self.classes,
                target_names=rotulos_rep,
                zero_division=0,
            )
        )
        return self

    def plotar(
        self,
        titulo: str = "Matriz de Confusão — rede35",
        normalizado: bool = False,
        ax: Axes | None = None,
    ) -> AvaliadorHopfield:
        """Plota a matriz de confusão como heatmap visual.

        Parameters
        ----------
        titulo : str, default="Matriz de Confusão — rede35"
            Título do gráfico.
        normalizado : bool, default=False
            Se True, normaliza as contagens por linha (percentual por classe real).
        ax : Axes | None, optional
            Eixo do Matplotlib; se None, instancia nova figura.

        Returns
        -------
        AvaliadorHopfield
            A própria instância.
        """
        if self.matriz_conf is None:
            raise RuntimeError(
                "[AvaliadorHopfield] Execute .avaliar() antes de .plotar()."
            )

        rotulos: list[str] = (
            self.nomes_classes if self.nomes_classes else [str(c) for c in self.classes]
        )

        mat: NDArray[Any] = self.matriz_conf.astype(float)
        fmt: str
        if normalizado:
            totais = mat.sum(axis=1, keepdims=True)
            totais[totais == 0] = 1
            mat = mat / totais
            fmt = ".1%"
        else:
            fmt = "d"
            mat = mat.astype(int)

        criar_figura: bool = ax is None
        current_ax: Axes
        if criar_figura:
            _, current_ax = plt.subplots(
                figsize=(max(6, len(self.classes)), max(5, len(self.classes)))
            )
        else:
            assert ax is not None
            current_ax = ax

        sns.heatmap(
            mat,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=rotulos,
            yticklabels=rotulos,
            ax=current_ax,
        )
        current_ax.set_xlabel("Predito")
        current_ax.set_ylabel("Real")
        current_ax.set_title(titulo)

        if criar_figura:
            plt.tight_layout()
            plt.show()
        return self

    def metricas_resumo(self, nome: str = "") -> dict[str, str | int | float]:
        """Retorna dicionário estruturado com as métricas calculadas.

        Parameters
        ----------
        nome : str, default=""
            Identificador do dataset analisado.

        Returns
        -------
        dict[str, Union[str, int, float]]
            Dicionário com métricas globais resumidas.
        """
        if self.acuracia is None or self.y_true is None:
            raise RuntimeError(
                "[AvaliadorHopfield] Execute .avaliar() antes de .metricas_resumo()."
            )
        return {
            "dataset": nome,
            "n_celulas": len(self.y_true),
            "acuracia": round(float(self.acuracia), 4),
            "f1_macro": round(float(self.f1_macro or 0.0), 4),
            "f1_weighted": round(float(self.f1_weighted or 0.0), 4),
            "taxa_reconstrucao": round(float(self.taxa_reconstrucao or 0.0), 4),
            "semelhanca_media": round(float(self.semelhanca_media or 0.0), 4),
        }

    def metricas_por_classe(self) -> pd.DataFrame:
        """Retorna DataFrame com precisão, revocação e pontuação F1 por classe.

        Returns
        -------
        pd.DataFrame
            DataFrame estruturado contendo precision, recall, f1 e contagem.
        """
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError(
                "[AvaliadorHopfield] Execute .avaliar() antes de .metricas_por_classe()."
            )
        rotulos: list[str] = (
            self.nomes_classes if self.nomes_classes else [str(c) for c in self.classes]
        )
        p, r, f, s = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            labels=self.classes,
            zero_division=0,
        )
        assert s is not None
        p_arr: NDArray[np.float64] = np.asarray(p, dtype=np.float64)
        r_arr: NDArray[np.float64] = np.asarray(r, dtype=np.float64)
        f_arr: NDArray[np.float64] = np.asarray(f, dtype=np.float64)
        s_arr: NDArray[np.int_] = np.asarray(s, dtype=int)
        return pd.DataFrame(
            {
                "classe_id": self.classes,
                "tipo_celular": rotulos,
                "precision": np.round(p_arr, 4),
                "recall": np.round(r_arr, 4),
                "f1_score": np.round(f_arr, 4),
                "support": s_arr,
                # Retrocompatibilidade
                "classe": rotulos,
                "n_celulas": s_arr,
                "precisao": np.round(p_arr, 4),
                "f1": np.round(f_arr, 4),
            }
        )

    def relatorio_classificacao_completo(self) -> pd.DataFrame:
        """Gera DataFrame estruturado com precision, recall, f1-score, support e médias globais.

        Returns
        -------
        pd.DataFrame
            DataFrame estruturado incluindo as linhas de classes e médias 'Macro Avg' e 'Weighted Avg'.
        """
        df_base: pd.DataFrame = self.metricas_por_classe()
        cols: list[str] = [
            "classe_id",
            "tipo_celular",
            "precision",
            "recall",
            "f1_score",
            "support",
        ]
        df_res: pd.DataFrame = df_base[cols].copy()
        tot_support: int = int(df_res["support"].sum())
        macro_p: float = round(float(df_res["precision"].mean()), 4)
        macro_r: float = round(float(df_res["recall"].mean()), 4)
        macro_f: float = round(float(df_res["f1_score"].mean()), 4)

        w_p: float
        w_r: float
        w_f: float
        if tot_support > 0:
            w_p = round(
                float(np.average(df_res["precision"], weights=df_res["support"])),
                4,
            )
            w_r = round(
                float(np.average(df_res["recall"], weights=df_res["support"])),
                4,
            )
            w_f = round(
                float(np.average(df_res["f1_score"], weights=df_res["support"])),
                4,
            )
        else:
            w_p, w_r, w_f = 0.0, 0.0, 0.0

        linhas_resumo: list[dict[str, Any]] = [
            {
                "classe_id": "—",
                "tipo_celular": "Macro Avg",
                "precision": macro_p,
                "recall": macro_r,
                "f1_score": macro_f,
                "support": tot_support,
            },
            {
                "classe_id": "—",
                "tipo_celular": "Weighted Avg",
                "precision": w_p,
                "recall": w_r,
                "f1_score": w_f,
                "support": tot_support,
            },
        ]
        return pd.concat([df_res, pd.DataFrame(linhas_resumo)], ignore_index=True)

    def __repr__(self) -> str:
        """Representação textual do avaliador Hopfield."""
        acc = (
            f"{self.acuracia * 100:.2f}%"
            if self.acuracia is not None
            else "não avaliado"
        )
        f1m = f"{self.f1_macro:.4f}" if self.f1_macro is not None else "—"
        f1w = f"{self.f1_weighted:.4f}" if self.f1_weighted is not None else "—"
        rec = (
            f"{self.taxa_reconstrucao * 100:.2f}%"
            if self.taxa_reconstrucao is not None
            else "—"
        )
        sim = (
            f"{self.semelhanca_media:.4f}" if self.semelhanca_media is not None else "—"
        )
        return (
            f"AvaliadorHopfield(\n"
            f"  padroes            = {self.padroes.shape}\n"
            f"  classes            = {self.classes}\n"
            f"  nc                 = {self.nc}\n"
            f"  metrica            = {self.metrica}\n"
            f"  acuracia           = {acc}\n"
            f"  f1_macro           = {f1m}\n"
            f"  f1_weighted        = {f1w}\n"
            f"  taxa_reconstrucao  = {rec}\n"
            f"  semelhanca_media   = {sim}\n"
            f")"
        )
