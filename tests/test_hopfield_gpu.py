import sys
from pathlib import Path
import numpy as np
import torch

# Adiciona raiz e src ao sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.hopfield import ModernHopfieldNetwork, closervects, wsort


def test_hopfield_cuda_inicializacao():
    rede = ModernHopfieldNetwork(beta=8.0, n_iters=1, binary=True)
    if torch.cuda.is_available():
        assert rede.device.type == "cuda"
    else:
        assert rede.device.type == "cpu"


def test_hopfield_store_retrieve_gpu():
    rede = ModernHopfieldNetwork(beta=10.0, n_iters=1, binary=True, threshold=0.0)

    # Cria 4 padrões binários distintos de 100 features
    padroes = np.array([
        [1.0] * 50 + [0.0] * 50,
        [0.0] * 50 + [1.0] * 50,
        [1.0, 0.0] * 50,
        [0.0, 1.0] * 50,
    ], dtype=np.float32)

    rede.store(padroes)
    assert rede.patterns.shape == (4, 100)
    assert rede.patterns.device.type == rede.device.type

    # Query: primeiro padrão com ruído sintético (30% dos bits invertidos)
    query = padroes[0].copy()
    query[10:25] = 0.0

    rec, prob = rede.retrieve(
        queries=query[None, :],
        batch_size=2,
        return_probabilities=True,
    )

    assert rec.shape == (1, 100)
    assert prob.shape == (1, 100)
    assert isinstance(rec, np.ndarray)
    assert isinstance(prob, np.ndarray)

    # Recuperou perfeitamente o padrão 0
    np.testing.assert_array_equal(rec[0], padroes[0])


def test_hopfield_salvar_carregar_portabilidade(tmp_path):
    rede = ModernHopfieldNetwork(beta=5.0, binary=True, threshold=0.2)
    padroes = np.eye(10, dtype=np.float32)
    rede.store(padroes)

    path_pt = tmp_path / "modelo_teste.pt"
    rede.salvar(path_pt)

    rede_carregada = ModernHopfieldNetwork.carregar(path_pt)
    assert rede_carregada.beta == 5.0
    assert rede_carregada.threshold == 0.2
    assert rede_carregada.patterns.shape == (10, 10)
    assert rede_carregada.device.type == rede.device.type


def test_hopfield_compute_attention_weights():
    rede = ModernHopfieldNetwork(beta=8.0, binary=True)
    padroes = np.eye(5, dtype=np.float32)
    rede.store(padroes)

    queries = padroes.copy()
    att = rede.compute_attention_weights(queries, batch_size=2)
    assert att.shape == (5, 5)
    # A atenção na diagonal principal deve ser dominante
    for i in range(5):
        assert np.argmax(att[i]) == i
