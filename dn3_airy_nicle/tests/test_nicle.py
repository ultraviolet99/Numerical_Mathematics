"""Testi za DN3.

scipy.special je uporabljen izključno kot orakelj za preverjanje; noben vgrajen
reševalec diferencialnih enačb v rešitvi ni uporabljen.
"""
import numpy as np
import pytest

from dn3_airy_nicle import A_airy, AI0, DAI0, ai, expm2, integriraj, magnus_korak, nicle
from dn3_airy_nicle.magnus import korak_dolzina

PRVIH_PET = [-2.338107410459, -4.087949444130, -5.520559828096,
             -6.786708090072, -7.944133587120]


# ---------- ze implementirano (zeleno takoj) ----------

def test_A_airy():
    np.testing.assert_allclose(A_airy(-3.5), [[0.0, 1.0], [-3.5, 0.0]])
    assert np.trace(A_airy(7.0)) == 0.0


def test_korak_dolzina_lastnosti():
    assert korak_dolzina(0.0, h0=0.05) <= 0.05
    x = -400.0
    assert korak_dolzina(x, h0=1.0, korakov_na_periodo=24.0) == pytest.approx(
        2 * np.pi / (24.0 * np.sqrt(1 + abs(x))), rel=1e-14)
    hs = [korak_dolzina(-t) for t in np.linspace(0, 300, 40)]
    assert np.all(np.diff(hs) <= 1e-15)                 # nenaraščajoč z globino


# ---------- expm2 ----------

def test_expm2_identiteta_in_determinanta():
    assert np.allclose(expm2(np.zeros((2, 2))), np.eye(2))
    sigma = np.array([[0.3, 1.2], [-0.7, -0.3]])        # brezsledna
    assert np.linalg.det(expm2(sigma)) == pytest.approx(1.0, abs=1e-14)


def test_expm2_rotacija_rocno():
    th = np.pi / 3
    E = expm2(np.array([[0.0, th], [-th, 0.0]]))        # z = -th^2 < 0
    np.testing.assert_allclose(E, [[0.5, np.sqrt(3) / 2],
                                   [-np.sqrt(3) / 2, 0.5]], atol=1e-15)


def test_expm2_hiperbolicna_in_diagonalna():
    th = 0.7
    E = expm2(np.array([[0.0, th], [th, 0.0]]))         # z = th^2 > 0
    np.testing.assert_allclose(E, [[np.cosh(th), np.sinh(th)],
                                   [np.sinh(th), np.cosh(th)]], atol=1e-15)
    a = 1.3
    np.testing.assert_allclose(expm2(np.diag([a, -a])), np.diag([np.e ** a, np.e ** -a]),
                               rtol=1e-14)


def test_expm2_proti_vrsti():
    sigma = np.array([[0.01, 0.05], [-0.02, -0.01]])
    vrsta, clen = np.eye(2), np.eye(2)
    for k in range(1, 25):
        clen = clen @ sigma / k
        vrsta = vrsta + clen
    assert np.allclose(expm2(sigma), vrsta, atol=1e-15)


def test_expm2_zveznost_vej():
    # z = +-1e-13: srednja (Taylorjeva) veja mora biti zvezna z zunanjima
    plus = expm2(np.array([[0.0, 1.0], [1e-13, 0.0]]))
    minus = expm2(np.array([[0.0, 1.0], [-1e-13, 0.0]]))
    np.testing.assert_allclose(plus, minus, atol=1e-11)


# ---------- magnus korak ----------

def test_magnus_proti_splosni_formuli():
    # zaprta oblika sigma (izpeljava v poročilu) mora ustrezati splošni formuli (18.37)
    x, h = -3.7, 0.05
    s3 = np.sqrt(3.0)
    A1 = A_airy(x + (0.5 - s3 / 6) * h)
    A2 = A_airy(x + (0.5 + s3 / 6) * h)
    sigma = h / 2 * (A1 + A2) - s3 / 12 * h ** 2 * (A1 @ A2 - A2 @ A1)
    y = np.array([0.3, -0.2])
    np.testing.assert_allclose(magnus_korak(x, y, h), expm2(sigma) @ y, rtol=1e-13)


def test_magnus_red4_empiricno():
    """Globalna napaka pri fiksnem x se mora manjšati kot h^4."""
    x_cilj, y0 = -1.0, np.array([AI0, DAI0])

    def resitev(h):
        x, y = 0.0, y0.copy()
        while x > x_cilj + 1e-15:
            hh = -min(h, x - x_cilj)
            y = magnus_korak(x, y, hh)
            x += hh
        return y[0]

    e1 = abs(resitev(0.02) - resitev(0.0025))
    e2 = abs(resitev(0.01) - resitev(0.0025))
    assert e1 / e2 == pytest.approx(16.0, rel=0.5)     # red 4: napaka ~ h^4


def test_wronskian_konstanten():
    # tr A = 0 -> Wronskijan dveh rešitev konstanten (det propagatorja = 1)
    xs, ya = integriraj(np.array([AI0, DAI0]), x_min=-20.0)
    xs2, yb = integriraj(np.array([1.0, 0.0]), x_min=-20.0)
    np.testing.assert_allclose(xs, xs2)
    W = ya[:, 0] * yb[:, 1] - ya[:, 1] * yb[:, 0]
    np.testing.assert_allclose(W, W[0], atol=1e-12)


# ---------- vrednosti in ničle ----------

def test_ai_proti_scipy():
    sc = pytest.importorskip("scipy.special")
    for x in [-1.0, -5.0, -15.0]:
        Ai, dAi = ai(x)
        ref = sc.airy(x)
        assert Ai == pytest.approx(ref[0], rel=1e-8)
        assert dAi == pytest.approx(ref[1], rel=1e-8)


def test_prvih_pet_nicel():
    np.testing.assert_allclose(nicle(5), PRVIH_PET, atol=1e-10)


def test_natancnost_dejansko_kvadraticna():
    """Newton mora doseči precej boljšo natančnost kot golo prepolavljanje
    oklepa: pri koraku h0 privzeto omejenem na ~800 korakov na nihaj je
    oklep širok reda 1e-2, 15 korakov bisekcije samih zato ne bi zadoščalo
    za 1e-10 (rel=0, da privzeti rtol numpyja napake ne prikrije)."""
    np.testing.assert_allclose(nicle(5), PRVIH_PET, atol=2e-11, rtol=0.0)


def test_nicle_brez_zgodnje_prekinitve():
    """Tudi če se konvergenčni kriterij nikoli ne sproži (tol nedosegljiv),
    Newton že prej pristane na fiksni točki, zato ostane natančnost enaka."""
    np.testing.assert_allclose(nicle(5, tol=-1.0), PRVIH_PET, atol=1e-10)


def test_nicle_na_intervalu():
    z = nicle(x_min=-10.0)                             # a1..a6 (a7 prib. -10.04 je zunaj)
    assert len(z) == 6
    assert z[-1] == pytest.approx(-9.02265085, abs=1e-7)


def test_urejenost_in_razmiki():
    z = nicle(20)
    assert np.all(np.diff(z) < 0)                      # padajoče
    razmiki = -np.diff(z)
    assert np.all(np.diff(razmiki) < 0)                # razmiki se ožijo


def test_alternacija_med_niclami():
    z = nicle(8)
    sredine = (z[:-1] + z[1:]) / 2
    vals = np.array([ai(m)[0] for m in sredine])
    assert np.all(vals != 0)
    assert np.all(np.sign(vals[:-1]) == -np.sign(vals[1:]))


def test_proti_scipy_oraklju():
    sc = pytest.importorskip("scipy.special")
    ref = sc.ai_zeros(30)[0]                           # prvih 30 ničel
    np.testing.assert_allclose(nicle(30), ref, atol=1e-10)


def test_mcmahon_asimptotika():
    z = nicle(40)
    k = np.arange(1, 41)
    t = 3 * np.pi * (4 * k - 1) / 8
    priblizek = -t ** (2 / 3) * (1 + 5 / 48 * t ** -2)
    # 2-clenska formula: rel napaka ~6e-4 pri k=1 in < 3e-7 za k >= 6 (mpmath);
    # male k pokrivata testa PRVIH_PET in scipy orakelj
    np.testing.assert_allclose(z[5:], priblizek[5:], rtol=2e-6)


# ---------- robni primeri in napake ----------

def test_ai_pozitiven_x_napaka():
    with pytest.raises(ValueError):
        ai(1.0)


def test_nicle_brez_n_in_x_min_napaka():
    with pytest.raises(ValueError):
        nicle()


def test_nicle_neveljaven_n_napaka():
    with pytest.raises(ValueError):
        nicle(n=0)
    with pytest.raises(ValueError):
        nicle(n=-3)


def test_nicle_neveljaven_x_min_napaka():
    with pytest.raises(ValueError):
        nicle(x_min=0.0)
    with pytest.raises(ValueError):
        nicle(x_min=5.0)


# --------- _precisti ----------

from dn3_airy_nicle.nicle import _precisti


a1 = nicle(1)[0]
assert _precisti(a1 - 0.02, a1 + 0.02) == pytest.approx(a1, abs=1e-11)


def test_precisti_bisekcijsko_varovalo():
    """Če Newtonov korak pade iz oklepa, mora varovalo preiti na bisekcijo.

    Funkcija f(x) = x ima ničlo v 0, a odvod ji zlažemo na zelo majhno
    vrednost, zato Newtonov korak vsakič skoči daleč iz oklepa. Postopek se
    mora izroditi v bisekcijo in kljub temu konvergirati.
    """
    klici = {"n": 0}

    def zlagan(x):
        klici["n"] += 1
        return x, 1e-12          # prava vrednost, absurdno majhen odvod

    x = _precisti(-1.0, 3.0, tol=1e-12, max_iter=200, f=zlagan)
    assert klici["n"] > 0
    assert abs(x) < 1e-12                      # bisekcija je vseeno našla ničlo
    assert -1.0 <= x <= 3.0                    # in ni ušla iz oklepa


def test_precisti_spostuje_max_iter():
    """Ob nedoseženem kriteriju se vrne zadnji približek, brez neskončne zanke."""
    x = _precisti(-1.0, 3.0, tol=1e-30, max_iter=3, f=lambda t: (t, 1e-12))
    assert -1.0 <= x <= 3.0