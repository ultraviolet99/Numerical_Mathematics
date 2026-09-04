"""Testi za DN2.

"""
import numpy as np
import pytest

from dn2_naravni_parameter import (cheb_koeficienti, cheb_tocke, clenshaw,
                                   gauss_legendre, hitrost, integral, s)

REL = 5e-11   # zahteva 18.2.1


def s_orakelj(t, dps=40):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = dps
    f = lambda tau: mp.sqrt(9 * tau**4 - 2 * tau**2 + 1)
    return float(mp.quad(f, [0, t]))


# ---------- hitrost  ----------

def test_hitrost_znane_vrednosti():
    assert hitrost(0.0) == pytest.approx(1.0, rel=1e-15)
    assert hitrost(1.0) == pytest.approx(np.sqrt(8.0), rel=1e-15)
    assert hitrost(np.sqrt(1 / 9)) == pytest.approx(np.sqrt(8 / 9), rel=1e-14)  # minimum
    np.testing.assert_allclose(hitrost([-2.0, 2.0]), [hitrost(2.0)] * 2)        # soda


# ---------- Gauss-Legendre----------

def test_gl_vozli_lastnosti():
    x, w = gauss_legendre(12)
    assert len(x) == len(w) == 12
    assert np.all(np.diff(x) > 0) and x[0] > -1 and x[-1] < 1
    np.testing.assert_allclose(x, -x[::-1], atol=1e-14)      # simetrija vozlov
    np.testing.assert_allclose(w, w[::-1], atol=1e-14)
    assert np.all(w > 0)
    assert np.sum(w) == pytest.approx(2.0, rel=1e-14)


def test_gl_polinomska_natancnost():
    # red n integrira polinome stopnje <= 2n-1 točno: n=5, x^9 na [0,1] -> 1/10
    assert integral(lambda x: x ** 9, 0.0, 1.0, 5) == pytest.approx(0.1, rel=1e-13)


def test_gl_na_znanem_integralu():
    assert integral(np.sin, 0.0, np.pi, 30) == pytest.approx(2.0, rel=1e-13)


def test_gl_en_vozel():
    # n = 1: edini vozel je x = 0, utež w = 2 (Newton se ustavi po prvem koraku)
    x, w = gauss_legendre(1)
    np.testing.assert_allclose(x, [0.0], atol=1e-15)
    np.testing.assert_allclose(w, [2.0], atol=1e-14)


def test_gl_predpomnjen():
    """Isti red se izračuna samo enkrat -- od tod O(1) cena vsakega klica s(t)."""
    gauss_legendre.cache_clear()
    prvi = gauss_legendre(45)
    drugi = gauss_legendre(45)
    assert prvi is drugi
    assert gauss_legendre.cache_info() == (1, 1, None, 1)


# ---------- Čebišev ----------

def test_clenshaw_rocno():
    # c = [1, 0, 1]: T0 + T2 = 1 + (2 xi^2 - 1) = 2 xi^2
    assert clenshaw(np.array([1.0, 0.0, 1.0]), 0.3) == pytest.approx(0.18, abs=1e-15)
    assert clenshaw(np.array([1.0, 0.0, 1.0]), -1.0) == pytest.approx(2.0, abs=1e-15)


def test_cheb_interpolacija_exp():
    m = 20
    u = cheb_tocke(m, 0.0, 1.0)
    c = cheb_koeficienti(np.exp(u))
    for uu in [0.0, 0.137, 0.5, 0.9, 1.0]:
        xi = 2 * uu - 1
        assert clenshaw(c, xi) == pytest.approx(np.exp(uu), abs=1e-13)


# ---------- s(t) ----------

def test_nicla_in_lihost():
    assert s(0.0) == 0.0
    for t in [0.3, 1.0, 7.5, 1e6]:
        assert s(-t) == pytest.approx(-s(t), rel=1e-14)


def test_nan_ostane_nan():
    assert np.isnan(s(float("nan")))


def test_celostevilski_vhod():
    # dokumentirano: t je lahko int ali float, rezultat je python float
    assert isinstance(s(2), float)
    assert s(2) == pytest.approx(s(2.0), rel=1e-14)


def test_taylor_okoli_nicle():
    # Taylorjev razvoj okoli 0: s(t) = t - t^3/3 + (4/5) t^5 + O(t^7)
    for t in [0.01, 0.03, 0.1]:
        priblizek = t - t ** 3 / 3 + 0.8 * t ** 5
        assert abs(s(t) - priblizek) < 2 * t ** 7


def test_aditivnost():
    assert s(2.0) == pytest.approx(s(1.0) + integral(hitrost, 1.0, 2.0, 60), rel=1e-12)


def test_monotonost():
    ts = np.linspace(-6, 6, 41)
    vals = [s(t) for t in ts]
    assert np.all(np.diff(vals) > 0)


def test_odvod_je_hitrost():
    for t in [0.5, 2.0, 4.0, 50.0]:
        h = 1e-5 * max(1.0, abs(t))
        num = (s(t + h) - s(t - h)) / (2 * h)
        assert num == pytest.approx(hitrost(t), rel=1e-7)


def test_zelo_veliki_t():
    # s(t) = t^3 (1 - 1/(3 t^2) + ...): pri t = 1e50 je popravek ~ 3e-101
    assert s(1e50) == pytest.approx(1e150, rel=1e-12)
    assert s(1e100) == pytest.approx(1e300, rel=1e-12)


# ---------- s(t): proti oraklju in prehod ----------

def test_proti_oraklju_mala_obmocja():
    for t in [1e-6, 0.1, 0.5, 1.0, 2.0, 2.999, 3.0]:
        assert s(t) == pytest.approx(s_orakelj(t), rel=REL)


def test_proti_oraklju_velika_obmocja():
    for t in [3.001, 5.0, 10.0, 100.0, 1e4, 1e8]:
        assert s(t) == pytest.approx(s_orakelj(t), rel=REL)


def test_zveznost_na_prehodu():
    eps = 1e-6   # dovolj velik, da razlika prevlada nad dovoljeno napako vej (5e-11 rel)
    assert s(3.0 + eps) - s(3.0 - eps) == pytest.approx(2 * eps * hitrost(3.0), rel=1e-3)


# ---------- predpomnjenje: cena klica ostane O(1) ----------

def test_h_koeficienti_predpomnjeni():
    """Čebiševi koeficienti (interpolacijska veja) se pripravijo natanko enkrat."""
    from dn2_naravni_parameter.param import _h_koeficienti

    _h_koeficienti.cache_clear()
    for t in (10.0, 1e3, 1e6, 1e9):
        s(t)
    assert _h_koeficienti.cache_info() == (3, 1, 1, 1)


# ---------- regresijske vrednosti (zamrznjeni parametri T0=3, N_GL=60, M_CHEB=40) ----------
# Reference izračunane enkrat z mpmath (dps=50); test jih ne rabi za vsak zagon

@pytest.mark.parametrize("t, vrednost", [
    (0.5, 0.48614380045653101),
    (1.0, 1.3577959303227702),
    (3.0, 26.794667249067505),
    (5.0, 124.14791154097491),
    (1000.0, 999999667.51077013),
    (1e8, 9.9999999999999997e23),
])
def test_regresijske_vrednosti(t, vrednost):
    assert s(t) == pytest.approx(vrednost, rel=REL)


# ---------- preliv v inf ----------

def test_preliv_vrne_inf():
    """Za |t| nad ~5.6e102 t^3 preseže obseg float64; pričakujemo inf, ne izjeme."""
    assert np.isinf(s(1e103))
    assert np.isinf(s(-1e103)) and s(-1e103) < 0