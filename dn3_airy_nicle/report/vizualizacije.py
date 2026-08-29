"""Demo skripta za DN3: ustvari slike in izpiše vse številke, ki nastopajo v poročilu.

Poganjaj iz korena paketa:

    python report/vizualizacije.py

Slike (PNG) nastanejo v report/slike/, številke pa se izpišejo na zaslon, da so
vse trditve iz poročila reproducibilne z enim ukazom. Slika napake rabi scipy
(orakelj); če ga ni na voljo, se ta del preskoči z obvestilom.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from dn3_airy_nicle import AI0, DAI0, A_airy, ai, integriraj, magnus_korak, nicle

MAPA = "report/slike"

# DLMF razdelek 9.9, prvih pet ničel Ai na 12 decimalk
PRVIH_PET = [-2.338107410459, -4.087949444130, -5.520559828096,
             -6.786708090072, -7.944133587120]


def slika_ai_z_niclami():
    xs, ys = integriraj(np.array([AI0, DAI0]), x_min=-12.0)
    z = nicle(x_min=-12.0)
    plt.figure(figsize=(5.4, 3.2))
    plt.plot(xs, ys[:, 0], lw=1.2)
    plt.plot(z, np.zeros_like(z), "o", color="tab:red", ms=4)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("x"); plt.ylabel("Ai(x)"); plt.title("Ai in njegove ničle")
    plt.tight_layout(); plt.savefig(f"{MAPA}/ai_nicle.png", dpi=200); plt.close()
    return z


def _resitev_pri(x_cilj, h):
    x, y = 0.0, np.array([AI0, DAI0])
    while x > x_cilj + 1e-15:
        hh = min(h, x - x_cilj)
        y = magnus_korak(x, y, -hh); x -= hh
    return y[0]


def slika_red4():
    hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    ref = _resitev_pri(-1.0, 0.002)
    err = np.array([abs(_resitev_pri(-1.0, h) - ref) for h in hs])
    plt.figure(figsize=(4.8, 3.2))
    plt.loglog(hs, err, "o-", label="izmerjeno")
    plt.loglog(hs, err[0] * (hs / hs[0]) ** 4, "k--", label=r"$\propto h^4$")
    plt.xlabel("h"); plt.ylabel("napaka pri x = -1"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{MAPA}/red4.png", dpi=200); plt.close()
    return hs, err


def _rk4_korak(x, y, h):
    k1 = A_airy(x) @ y
    k2 = A_airy(x + h / 2) @ (y + h / 2 * k1)
    k3 = A_airy(x + h / 2) @ (y + h / 2 * k2)
    k4 = A_airy(x + h) @ (y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def slika_wronskian():
    """Strukturni invariant: Magnus ga ohranja, RK4 ne."""
    h, x_min = 0.05, -60.0
    ya, yb = np.array([AI0, DAI0]), np.array([1.0, 0.0])
    za = {"Magnus": [ya.copy()], "RK4": [ya.copy()]}
    zb = {"Magnus": [yb.copy()], "RK4": [yb.copy()]}
    xs = [0.0]
    x = 0.0
    while x > x_min + 1e-12:
        hh = min(h, x - x_min)
        za["Magnus"].append(magnus_korak(x, za["Magnus"][-1], -hh))
        zb["Magnus"].append(magnus_korak(x, zb["Magnus"][-1], -hh))
        za["RK4"].append(_rk4_korak(x, za["RK4"][-1], -hh))
        zb["RK4"].append(_rk4_korak(x, zb["RK4"][-1], -hh))
        x -= hh; xs.append(x)
    xs = np.array(xs)
    drift_koncni = {}
    plt.figure(figsize=(5.2, 3.2))
    for ime, sl in [("Magnus", "-"), ("RK4", "--")]:
        A = np.array(za[ime]); B = np.array(zb[ime])
        W = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        drift_koncni[ime] = abs(W[-1] - W[0])
        plt.semilogy(xs, np.maximum(np.abs(W - W[0]), 1e-18), sl, label=ime)
    plt.xlabel("x"); plt.ylabel(r"$|W(x) - W(0)|$"); plt.legend()
    plt.title("Drift Wronskijana (h = 0.05)")
    plt.tight_layout(); plt.savefig(f"{MAPA}/wronskian.png", dpi=200); plt.close()
    return drift_koncni


def slika_napaka_nicel(n=60, m=800.0):
    """|naša - scipy| vs k + skalirni zakon C''|a_k| m^(-4)."""
    from scipy import special
    z = np.asarray(nicle(n))
    ref = special.ai_zeros(n)[0]
    d = np.abs(z - ref)
    C = np.median(d / np.abs(ref)) * m ** 4
    plt.figure(figsize=(5.0, 3.2))
    k = np.arange(1, n + 1)
    plt.semilogy(k, np.maximum(d, 1e-16), "o", ms=3, label="izmerjeno")
    plt.semilogy(k, C * np.abs(ref) * m ** -4, "k--",
                 label=r"$C''\,|a_k|\,m^{-4}$")
    plt.axhline(1e-10, color="r", ls=":", label="cilj 1e-10")
    plt.xlabel("k"); plt.ylabel(r"$|\Delta a_k|$"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{MAPA}/napaka_nicel.png", dpi=200); plt.close()
    return z, ref, d, C



# ============================ številke ============================

def izpis_prvih_pet(z):
    print("\n--- Prvih pet ničel proti DLMF ---")
    print(f"{'k':>3s} {'naša':>18s} {'DLMF':>18s} {'razlika':>10s}")
    for k, (naša, dlmf) in enumerate(zip(z, PRVIH_PET), start=1):
        print(f"{k:3d} {naša:18.12f} {dlmf:18.12f} {naša - dlmf:10.1e}")


def izpis_red4(hs, err):
    print("\n--- Empirični red Magnusove metode (x0=0 -> x=-1) ---")
    for h, e in zip(hs, err):
        print(f"  h = {h:7.4f}  napaka = {e:.3e}")
    naklon = np.polyfit(np.log(hs), np.log(err), 1)[0]
    print(f"  naklon log-log (pričakovano 4): {naklon:.2f}")


def izpis_wronskian(drift):
    print("\n--- Drift Wronskijana na [-60, 0] (h = 0.05) ---")
    for ime, d in drift.items():
        print(f"  {ime:>6s}: |W(x_min) - W(0)| = {d:.3e}")
    print(f"  razmerje RK4/Magnus: {drift['RK4'] / max(drift['Magnus'], 1e-300):.1e}")


def izpis_napaka_nicel(z, ref, d, C):
    print("\n--- Napaka ničel proti scipy (n = 60, m = 800) ---")
    i = int(np.argmax(d))
    print(f"  največja napaka = {d[i]:.2e} pri k = {i + 1} (a_k = {ref[i]:.4f})")
    print(f"  umerjeni C'' (iz C''|a_k|m^-4): {C:.2e}")
    print(f"  vseh {len(d)} ničel < 1e-10: {bool(np.all(d < 1e-10))}")
    # napoved: pri istem m, do katere globine |a_k| ostane napaka < 1e-10
    a_meja = 1e-10 / (C * 800.0 ** -4)
    print(f"  napovedana meja |a_k| < {a_meja:.0f} za napako < 1e-10 pri m = 800")


def izpis_cas():
    print("\n--- Čas izračuna ---")
    for n in (30, 100, 300):
        z = time.perf_counter()
        nicle(n)
        t = time.perf_counter() - z
        print(f"  nicle({n:4d}): {t * 1000:7.1f} ms  ({t / n * 1e6:6.1f} µs/ničlo)")


if __name__ == "__main__":
    z = slika_ai_z_niclami()
    hs, err = slika_red4()
    drift = slika_wronskian()
    print("slike shranjene v", MAPA)

    izpis_prvih_pet(nicle(5))
    izpis_red4(hs, err)
    izpis_wronskian(drift)
    izpis_cas()

    try:
        podatki = slika_napaka_nicel()
        izpis_napaka_nicel(*podatki)
    except ImportError as e:
        print("PRESKOČENO (manjka paket):", e)
