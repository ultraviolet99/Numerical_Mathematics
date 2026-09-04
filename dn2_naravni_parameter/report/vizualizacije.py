"""Demo skripta za DN2: ustvari slike in izpiše vse številke, ki nastopajo v poročilu.

Poganjaj iz korena paketa:

    python report/vizualizacije.py

Slike (PNG) nastanejo v report/slike/, številke pa se izpišejo na zaslon, da so
vse trditve iz poročila reproducibilne z enim ukazom. Slika napake in izpisi,
ki potrebujejo orakelj, rabijo mpmath; če ga ni na voljo, se le-ti preskočijo.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from dn2_naravni_parameter import (cheb_koeficienti, cheb_tocke, hitrost,
                                   integral, s)
from dn2_naravni_parameter.param import M_CHEB, N_GL, T0, _h_koeficienti

MAPA = "report/slike"
TAU_S = np.array([np.sqrt(2) + 1j, np.sqrt(2) - 1j,
                  -np.sqrt(2) + 1j, -np.sqrt(2) - 1j]) / 3   # razvejišča (izpeljava v poročilu)


def _t_pri_s(cilj, t_hi=20.0):
    """Inverz: bisekcija za s(t) = cilj (s je naraščajoča)."""
    lo, hi = 0.0, t_hi
    for _ in range(80):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if s(mid) < cilj else (lo, mid)
    return (lo + hi) / 2


# =============================== slike ===============================

def slika_krivulja():
    """Krivulja s pikami na enakomernih ločnih razdaljah - kaj s(t) sploh je."""
    t = np.linspace(-1.6, 1.6, 400)
    plt.figure(figsize=(4.6, 3.6))
    plt.plot(t**3 - t, t**2 - 1, lw=1.2)
    cilji = np.arange(1, 14) * (2 * s(1.6) / 14) - s(1.6)
    tt = np.array([np.sign(c) * _t_pri_s(abs(c), 1.6) for c in cilji])
    plt.plot(tt**3 - tt, tt**2 - 1, "o", ms=4, color="tab:red")
    plt.title("Točke na enakomernih ločnih razdaljah")
    plt.gca().set_aspect("equal")
    plt.tight_layout(); plt.savefig(f"{MAPA}/krivulja.png", dpi=200); plt.close()


def slika_konvergenca_gl():
    """Napaka GL za s(3) v odvisnosti od n + teoretični naklon rho^(-2n)."""
    ref = integral(hitrost, 0.0, 3.0, 200)
    ni = np.arange(4, 80, 4)
    err = np.array([abs(integral(hitrost, 0.0, 3.0, int(n)) - ref) / ref for n in ni])
    plt.figure(figsize=(4.8, 3.2))
    plt.semilogy(ni, np.maximum(err, 1e-17), "o-", label="izmerjeno")
    rho = 1.34
    plt.semilogy(ni, err[0] * rho ** (-2 * (ni - ni[0])), "k--",
                 label=r"$\propto \rho^{-2n},\ \rho=1.34$")
    plt.axvline(N_GL, color="tab:gray", ls=":", label=f"$n = {N_GL}$")
    plt.xlabel("n"); plt.ylabel("rel. napaka")
    plt.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.ylim(1e-20, 1e2)
    plt.tight_layout(); plt.savefig(f"{MAPA}/konv_gl.png", dpi=200); plt.close()
    return ni, err


def slika_cheb_koef():
    """|c_k| s platojem pri strojni natančnosti - utemeljitev M_CHEB."""
    c = np.abs(np.asarray(_h_koeficienti(), float))
    plt.figure(figsize=(4.8, 3.2))
    plt.semilogy(np.maximum(c, 1e-18), "o-", label=r"$|c_k|$")
    k = np.arange(len(c))
    plt.semilogy(k, c[0] * 19.0 ** (-k.astype(float)), "k--", label=r"$\propto 19^{-k}$")
    plt.xlabel("k"); plt.legend()
    plt.ylim(1e-20, 1e2)
    plt.tight_layout(); plt.savefig(f"{MAPA}/cheb_koef.png", dpi=200); plt.close()
    return c


def slika_singularnosti():
    """Razvejišča in Bernsteinova elipsa za [0, T_0]."""
    plt.figure(figsize=(5.2, 3.2))
    th = np.linspace(0, 2 * np.pi, 400)
    for rho, sl in [(1.34, "-")]:
        xi = (rho * np.exp(1j * th) + np.exp(-1j * th) / rho) / 2
        tau = T0 / 2 + T0 / 2 * xi
        plt.plot(tau.real, tau.imag, "k" + sl, lw=1, label=fr"$E_\rho,\ \rho={rho}$")
    plt.plot([0, T0], [0, 0], lw=3, color="tab:blue", label="[0, $T_0$]")
    plt.plot(TAU_S.real, TAU_S.imag, "rx", ms=8, label="razvejišča")
    plt.gca().set_aspect("equal")
    plt.ylim(-0.8, 0.8)
    plt.title(r"Analitičnost integranda v $\tau$-ravnini")
    plt.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout(); plt.savefig(f"{MAPA}/singularnosti.png", dpi=200); plt.close()


def slika_napaka():
    """Rel. napaka proti mpmath oraklju čez obe veji."""
    import mpmath as mp
    mp.mp.dps = 40
    g = lambda tau: mp.sqrt(9 * tau**4 - 2 * tau**2 + 1)
    ts = np.logspace(-2, 8, 33)
    rel = []
    for t in ts:
        ref = mp.quad(g, [0, mp.mpf(t)])
        rel.append(abs(s(float(t)) - float(ref)) / float(ref))
    rel = np.array(rel)
    plt.figure(figsize=(4.8, 3.2))
    plt.loglog(ts, np.maximum(rel, 1e-17), "o-")
    plt.axhline(5e-11, color="r", ls="--", label="zahteva 5e-11")
    plt.axvline(T0, color="k", ls=":", label="prehod $T_0$")
    plt.xlabel("t"); plt.ylabel("rel. napaka")
    plt.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout(); plt.savefig(f"{MAPA}/napaka.png", dpi=200); plt.close()
    return ts, rel


def slika_cas(ponovitev=5, klicev=200):
    """Čas na klic je neodvisen od t (zahteva O(1)).
 
    Vsako meritev ponovimo večkrat in vzamemo najkrajši čas: motnje sistema
    čas lahko le podaljšajo, zato je
    najkrajši izmed vzorcev najbližji dejanski ceni klica.
    """
    s(1.0); s(10.0)                                    # ogrej predpomnilnike
    ts = [1.0, 1e2, 1e4, 1e6, 1e9, 1e12]
    case = []
    for t in ts:
        vzorci = []
        for _ in range(ponovitev):
            z = time.perf_counter()
            for _ in range(klicev):
                s(t)
            vzorci.append((time.perf_counter() - z) / klicev)
        case.append(min(vzorci))
    case = np.array(case)
    plt.figure(figsize=(4.8, 3.0))
    plt.semilogx(ts, case * 1e6, "o-")
    plt.xlabel("t"); plt.ylabel("čas na klic [v µs]")
    plt.ylim(0, case.max() * 1e6 * 1.6)
    plt.tight_layout(); plt.savefig(f"{MAPA}/cas.png", dpi=200); plt.close()
    return ts, case



# ============================ številke ============================

def izpis_konvergenca_gl(ni, err):
    """Kje napaka GL kvadrature doseže plato in kako se ujema s teorijo."""
    print("\n--- Konvergenca GL na [0, T0] (referenca: n = 200) ---")
    for n, e in zip(ni[::3], err[::3]):
        print(f"  n = {n:3d}   rel. napaka = {e:.2e}")
    plato = ni[np.argmax(err < 1e-15)] if np.any(err < 1e-15) else None
    print(f"  plato (< 1e-15) dosežen pri n = {plato}; izbrano N_GL = {N_GL}")


def izpis_cheb_koef(c):
    """Kje |c_k| doseže plato in kako se ujema z oceno rho_u 19."""
    print("\n--- Čebiševi koeficienti h(u) ---")
    for k in (0, 2, 4, 6, 8, 10, 12, 20, M_CHEB):
        print(f"  |c_{k:2d}| = {c[k]:.3e}")
    plato = int(np.argmax(c < 1e-15)) if np.any(c < 1e-15) else None
    print(f"  plato (< 1e-15) dosežen pri k = {plato}; izbrano M_CHEB = {M_CHEB}")


def izpis_napaka(ts, rel):
    """Največja izmerjena napaka proti oraklju in njena lokacija."""
    print("\n--- Napaka proti mpmath oraklju (33 točk, t od 1e-2 do 1e8) ---")
    i = int(np.argmax(rel))
    print(f"  največja rel. napaka = {rel[i]:.2e} pri t = {ts[i]:.3g}")
    print(f"  vseh {len(rel)} točk < 5e-11: {bool(np.all(rel < 5e-11))}")


def izpis_prehod():
    """Ujemanje vej tik pred in tik po T0."""
    eps = 1e-6
    leva, desna = s(T0 - eps), s(T0 + eps)
    print("\n--- Prehod pri T0 ---")
    print(f"  s(T0 - {eps:g}) = {leva:.10f}")
    print(f"  s(T0 + {eps:g}) = {desna:.10f}")
    print(f"  razlika = {desna - leva:.3e},"
          f" pričakovano 2*eps*hitrost(T0) = {2 * eps * hitrost(T0):.3e}")


def izpis_cas(ts, case):
    """Časi klica s(t): potrditev O(1) ne glede na velikost t."""
    print("\n--- Čas na klic (najkrajši od 5 ponovitev po 200 klicev) ---")
    for t, c in zip(ts, case):
        print(f"  t = {t:<9.0e}  {c * 1e6:6.2f} µs")
    print(f"  razmerje najpočasnejši/najhitrejši klic: {case.max() / case.min():.2f}")


def izpis_regresija():
    """Tabela vrednosti s(t), ki se uporabljajo tudi kot regresijske vrednosti v testih."""
    print("\n--- Vrednosti s(t) (12 mest, T0 = 3, N_GL = 60, M_CHEB = 40) ---")
    for t in (0.5, 1.0, 3.0, 5.0, 1000.0, 1e8):
        print(f"  s({t:<10g}) = {s(t):.12g}")


if __name__ == "__main__":
    slika_krivulja()
    ni, err = slika_konvergenca_gl()
    c = slika_cheb_koef()
    slika_singularnosti()
    izpis_regresija()
    izpis_konvergenca_gl(ni, err)
    izpis_cheb_koef(c)
    izpis_prehod()
    ts_cas, case = slika_cas()
    izpis_cas(ts_cas, case)
    print(f"\nslike shranjene v {MAPA}/")

    try:
        ts, rel = slika_napaka()
        izpis_napaka(ts, rel)
    except ImportError as e:
        print("PRESKOČENO (manjka paket):", e)
