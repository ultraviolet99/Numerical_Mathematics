"""Testi za DN1.

"""
import numpy as np
import pytest

from dn1_qr_premik import (EnojniPremik, WilkinsonovPremik, eigen, givens, laplaceova_matrika,
                           podobnostna_matrika, qr_korak_premik, tridiag)


# ---------- pomožne --------

def laplace_1d(n):
    """Matrika 1D Laplaceovega operatorja: 2 na diagonali, -1 ob njej."""
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def tridiag_matrika(d, e):
    return np.diag(d) + np.diag(e, 1) + np.diag(e, -1)


def det_tridiag(d, e):
    """Determinanta simetrične tridiagonalne prek kontinuante."""
    f_prev, f = 1.0, d[0]
    for i in range(1, len(d)):
        f_prev, f = f, d[i] * f - e[i - 1] ** 2 * f_prev
    return f


# ---------- givens ---------

def test_givens_lastnosti():
    for a, b in [(3.0, 4.0), (-2.0, 0.5), (0.0, 1.0), (1e200, 1e200), (1.0, 0.0)]:
        c, s, r = givens(a, b)
        assert c * c + s * s == pytest.approx(1.0, abs=1e-15)
        assert -s * a + c * b == pytest.approx(0.0, abs=1e-12 * max(abs(a), abs(b), 1))
        assert c * a + s * b == pytest.approx(r, rel=1e-14)
        assert np.isfinite(r)                      # brez prekoračitve pri 1e200
    assert givens(5.0, 0.0)[:2] == (1.0, 0.0)


# --------- en QR korak ----------

def test_qr_korak_rocno_2x2():
    # T = [[3,1],[1,1]], mu = 1 -> d' = (17/5, 3/5), |e'| = 1/5
    d, e = qr_korak_premik(np.array([3.0, 1.0]), np.array([1.0]), 1.0)
    np.testing.assert_allclose(d, [17 / 5, 3 / 5], atol=1e-14)
    assert abs(e[0]) == pytest.approx(1 / 5, abs=1e-14)


def test_qr_korak_invarianti():
    rng = np.random.default_rng(3)
    d0, e0 = rng.standard_normal(12), rng.standard_normal(11)
    d, e = qr_korak_premik(d0.copy(), e0.copy(), d0[-1])
    assert np.sum(d) == pytest.approx(np.sum(d0), rel=1e-12)          # sled
    assert det_tridiag(d, e) == pytest.approx(det_tridiag(d0, e0), rel=1e-9)


def test_qr_korak_stagnacija_enojnega_premika():
    #[[2,1],[1,2]] z mu = 2 se ne spremeni -> potreba po varovalu
    d, e = qr_korak_premik(np.array([2.0, 2.0]), np.array([1.0]), 2.0)
    np.testing.assert_allclose(d, [2.0, 2.0], atol=1e-14)
    assert abs(e[0]) == pytest.approx(1.0, abs=1e-14)


# ---------- tridiagonalizacija ----------

def test_tridiag_rekonstrukcija():
    rng = np.random.default_rng(11)
    B = rng.standard_normal((8, 8))
    A = (B + B.T) / 2
    d, e, U = tridiag(A)
    np.testing.assert_allclose(U.T @ U, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(U @ tridiag_matrika(d, e) @ U.T, A, atol=1e-11)


def test_tridiag_ze_tridiagonalne():
    d0, e0 = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5])
    d, e, _ = tridiag(tridiag_matrika(d0, e0))
    np.testing.assert_allclose(d, d0, atol=1e-13)
    np.testing.assert_allclose(np.abs(e), np.abs(e0), atol=1e-13)   # predznak ni določen


# ---------- eigen ----------

def test_rocno_2x2():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])           # lastni vrednosti 1 in 3
    np.testing.assert_allclose(eigen(A, EnojniPremik()), [1.0, 3.0], atol=1e-12)


def test_antidiagonalna_zahteva_varovalo():
    # enojni premik tu stagnira; eigen mora vseeno vrniti [-1, 1]
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(eigen(A, EnojniPremik()), [-1.0, 1.0], atol=1e-12)


def test_laplace_1d_analiticno():
    n = 30
    k = np.arange(1, n + 1)
    tocne = 2 - 2 * np.cos(k * np.pi / (n + 1))       # znana rešitev
    dobljene = eigen(laplace_1d(n), EnojniPremik())
    np.testing.assert_allclose(np.sort(dobljene), np.sort(tocne), atol=1e-10)


def test_proti_numpy_oraklju():
    rng = np.random.default_rng(42)
    B = rng.standard_normal((40, 40))
    A = (B + B.T) / 2
    np.testing.assert_allclose(eigen(A, EnojniPremik()),
                               np.linalg.eigvalsh(A), atol=1e-9)


def test_vektorji_lastnosti():
    rng = np.random.default_rng(7)
    B = rng.standard_normal((25, 25))
    A = (B + B.T) / 2
    lam, V = eigen(A, EnojniPremik(), vektorji=True)
    np.testing.assert_allclose(V.T @ V, np.eye(25), atol=1e-9)          # ortogonalnost
    np.testing.assert_allclose(A @ V, V @ np.diag(lam), atol=1e-8)      # A V = V Λ


def test_robni_primeri():
    np.testing.assert_allclose(eigen(np.array([[5.0]]), EnojniPremik()), [5.0])
    D = np.diag([3.0, 1.0, 2.0])                       # že diagonalna
    np.testing.assert_allclose(eigen(D, EnojniPremik()), [1.0, 2.0, 3.0], atol=1e-12)
    A = np.diag([1.0, 1.0, 2.0])                       # večkratna lastna vrednost
    lam, V = eigen(A, EnojniPremik(), vektorji=True)
    np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-10)


# ---------- graf podobnosti -----

def test_podobnostna_matrika_lastnosti():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])            # razdalja 5
    W = podobnostna_matrika(X, sigma=1.0)
    assert W.shape == (2, 2)
    np.testing.assert_allclose(np.diag(W), 0.0)
    np.testing.assert_allclose(W, W.T)
    assert W[0, 1] == pytest.approx(np.exp(-12.5), rel=1e-12)


def test_laplaceova_matrika_grafa():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (10, 2)), rng.normal(3, 0.3, (12, 2))])
    L = laplaceova_matrika(podobnostna_matrika(X, sigma=0.8))
    lam = eigen(L, EnojniPremik())
    assert abs(lam[0]) < 1e-8                          # (konstanten vektor)
    assert lam[1] < 0.5                                # šibka povezava med gručama


def test_fiedlerjev_razrez():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 0.25, (10, 2)), rng.normal(4, 0.25, (12, 2))])
    L = laplaceova_matrika(podobnostna_matrika(X, sigma=1.0))
    _, V = eigen(L, EnojniPremik(), vektorji=True)
    f = V[:, 1]                                        # Fiedlerjev vektor
    g1, g2 = f[:10], f[10:]
    assert (np.all(g1 > 0) and np.all(g2 < 0)) or (np.all(g1 < 0) and np.all(g2 > 0))


# ---------- robni primeri in napake ---------
 
def test_qr_korak_1x1():
    """Za 1x1 je T - mu*I = [d0-mu], Q = [1], torej RQ + mu*I = T: nespremenjeno."""
    d, e = qr_korak_premik(np.array([5.0]), np.array([]), 2.0)
    np.testing.assert_allclose(d, [5.0], atol=1e-15)
    assert len(e) == 0
 
 
def test_nekvadratna_matrika():
    with pytest.raises(ValueError):
        eigen(np.ones((2, 3)), EnojniPremik())
    with pytest.raises(ValueError):
        tridiag(np.ones((2, 3)))
 
 
def test_nesimetricna_matrika():
    """Postopek predpostavlja simetrijo, zato mora nesimetričen vhod javiti napako."""
    with pytest.raises(ValueError):
        eigen(np.array([[1.0, 2.0], [3.0, 4.0]]), EnojniPremik())
 
 
def test_prekoracitev_iteracij():
    """Premajhen max_iter mora javiti RuntimeError, ne pa vrniti napačnega rezultata."""
    rng = np.random.default_rng(9)
    B = rng.standard_normal((20, 20))
    A = (B + B.T) / 2
    with pytest.raises(RuntimeError):
        eigen(A, EnojniPremik(), max_iter=3)
 
 
def test_metoda_izbere_premik():
    """Argument metoda ni okras: enojni premik mora biti dejansko uporabljen."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(eigen(A, EnojniPremik()), [1.0, 3.0], atol=1e-12)
 
 
# --------- konvergenca ----------
 
def test_kubicna_konvergenca():
    """Ena iteracija zmanjša |e| priblizno s tretjo potenco"""
    napake = []
    for eps in (1e-3, 1e-4):
        d = np.array([2.7, -1.3, 0.9, 3.4])
        e = np.array([0.8, 0.7, eps])
        _, e_nov = qr_korak_premik(d.copy(), e.copy(), d[-1])
        napake.append(abs(e_nov[-1]))
    p = np.log(napake[0] / napake[1]) / np.log(1e-3 / 1e-4)
    assert 2.5 < p < 3.5
 
 
def test_stevilo_korakov_na_lastno_vrednost():
    """Kubična konvergencija pomeni O(1) korakov na lastno vrednost (tu < 5)."""
    import sys
    modul = sys.modules["dn1_qr_premik.eigen"]
    izvirna = modul.qr_korak_premik
    stevec = {"n": 0}
 
    def steti(d, e, mu, V=None):
        stevec["n"] += 1
        return izvirna(d, e, mu, V)
 
    modul.qr_korak_premik = steti
    try:
        rng = np.random.default_rng(1)
        B = rng.standard_normal((40, 40))
        modul.eigen((B + B.T) / 2, EnojniPremik())
    finally:
        modul.qr_korak_premik = izvirna
    assert stevec["n"] / 40 < 5.0
 
 
# ---------- strategija premika (argument metoda) ---------
 
def test_metoda_res_doloci_premik():
    """eigen mora premik dobiti od strategije, ne imeti zakodiranega."""
    from dn1_qr_premik import EnojniPremik as _EP
 
    class StejocPremik(_EP):
        def __init__(self):
            self.klici = 0
 
        def premik(self, d, e, lo, hi):
            self.klici += 1
            return super().premik(d, e, lo, hi)
 
    m = StejocPremik()
    rng = np.random.default_rng(4)
    B = rng.standard_normal((15, 15))
    A = (B + B.T) / 2
    np.testing.assert_allclose(eigen(A, m), np.linalg.eigvalsh(A), atol=1e-10)
    assert m.klici > 0
 
 
def test_wilkinsonova_strategija():
    """Druga strategija mora dati enake lastne vrednosti po drugi poti."""
    from dn1_qr_premik import WilkinsonovPremik
    rng = np.random.default_rng(6)
    B = rng.standard_normal((20, 20))
    A = (B + B.T) / 2
    np.testing.assert_allclose(eigen(A, WilkinsonovPremik()),
                               np.linalg.eigvalsh(A), atol=1e-10)
 
 
def test_neznana_strategija():
    with pytest.raises(ValueError):
        eigen(np.eye(3), metoda="enojni")
 
 
def test_privzeta_strategija():
    """Brez argumenta metoda se uporabi enojni premik."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(eigen(A), [1.0, 3.0], atol=1e-12)

# -------------------

def test_podan_tol():
    """Eksplicitno podan prag mora biti upoštevan."""
    A = np.array([[2.0, 1e-9], [1e-9, 3.0]])
    np.testing.assert_allclose(eigen(A, EnojniPremik(), tol=1e-6), [2.0, 3.0], atol=1e-12)
