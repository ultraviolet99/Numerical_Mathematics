"""Demo skripta za DN1: ustvari slike in izpiše vse številke, ki nastopajo v poročilu.

Poganjaj iz korena paketa:

    python report/vizualizacije.py

Slike (PNG) nastanejo v report/slike/, številke pa se izpišejo na zaslon, da so
vse trditve iz poročila reproducibilne z enim ukazom. Ker so razlike na ravni
zadnjih bitov odvisne od izvedbe BLAS, se lahko števila korakov in velikosti
napak na drugi napravi razlikujejo za nekaj odstotkov; zato skripta izpiše tudi
različico numpy.
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from dn1_qr_premik import (EnojniPremik, eigen, laplaceova_matrika,
                           podobnostna_matrika, qr_korak_premik, tridiag)

MAPA = "report/slike"


def _simetricna(rng, n):
    """Naključna simetrična matrika velikosti n x n."""
    B = rng.standard_normal((n, n))
    return (B + B.T) / 2


def _podatki_gruc(rng, n1=60, n2=70):
    """Dva Gaussova oblaka točk v ravnini (skupaj n1 + n2 točk)."""
    return np.vstack([rng.normal([0, 0], 0.45, (n1, 2)),
                      rng.normal([3.0, 1.5], 0.45, (n2, 2))])


# =============================== slike ===============================

def slika_konvergenca():
    """Semilog |e_zadnji| po korakih: kubični padec tik pred deflacijo."""
    rng = np.random.default_rng(1)
    d, e, _ = tridiag(_simetricna(rng, 12))
    zgod = [abs(e[-1])]
    for _ in range(30):
        d, e = qr_korak_premik(d, e, d[-1])
        zgod.append(abs(e[-1]))
        if zgod[-1] < 1e-15:
            break
    plt.figure(figsize=(5, 3.2))
    plt.semilogy(zgod, "o-")
    plt.xlabel("korak"); plt.ylabel(r"$|e_{m-1}|$")
    plt.title("Konvergenca zadnjega obdiagonalnega elementa")
    plt.tight_layout(); plt.savefig(f"{MAPA}/konvergenca.png", dpi=200); plt.close()
    return zgod


def slika_gruce():
    """Točke, pobarvane po predznaku Fiedlerjevega vektorja."""
    rng = np.random.default_rng(2)
    X = _podatki_gruc(rng)
    L = laplaceova_matrika(podobnostna_matrika(X, sigma=0.8))
    _, V = eigen(L, EnojniPremik(), vektorji=True)
    barve = np.where(V[:, 1] >= 0, "tab:blue", "tab:red")
    plt.figure(figsize=(4.6, 3.6))
    plt.scatter(X[:, 0], X[:, 1], c=barve, s=18)
    plt.title("Spektralni razrez (predznak $V[:,1]$)")
    plt.gca().set_aspect("equal")
    plt.tight_layout(); plt.savefig(f"{MAPA}/gruce.png", dpi=200); plt.close()


def slika_spekter():
    """Prvih 10 lastnih vrednosti L: vrzel v spektru za dve gruči."""
    rng = np.random.default_rng(2)
    L = laplaceova_matrika(podobnostna_matrika(_podatki_gruc(rng), sigma=0.8))
    lam = eigen(L, EnojniPremik())
    plt.figure(figsize=(4.6, 3.0))
    plt.plot(np.arange(1, 11), lam[:10], "o")
    plt.xlabel("k"); plt.ylabel(r"$\lambda_k$"); plt.title("Spodnji del spektra $L$")
    plt.tight_layout(); plt.savefig(f"{MAPA}/spekter.png", dpi=200); plt.close()


def slika_casi(ni=(25, 50, 100, 200), ponovitev=3):
    """Log-log čas v odvisnosti od n; izmerjeni naklon je približno 2.

    Vsako meritev ponovimo večkrat in vzamemo najkrajši čas, ker motnje sistema
    čas lahko le podaljšajo.

    Redukcija na tridiagonalno obliko je sicer O(n^3), a je izvedena z
    matričnimi operacijami knjižnice numpy; prevladuje iteracijski del, ki je
    napisan kot navadna zanka in je O(n^2). Zato je referenčna črta n^2.
    """
    t_moj, t_np = [], []
    rng = np.random.default_rng(3)
    ogrevanje = _simetricna(np.random.default_rng(0), 30)   # prva meritev ni popačena
    eigen(ogrevanje, EnojniPremik()); np.linalg.eigvalsh(ogrevanje)
    for n in ni:
        A = _simetricna(rng, n)
        # najkrajši od več zagonov: motnje (razporejevalnik, GC) čas le podaljšajo
        cas_moj, cas_np = [], []
        for _ in range(ponovitev):
            z = time.perf_counter(); eigen(A, EnojniPremik()); cas_moj.append(time.perf_counter() - z)
            z = time.perf_counter(); np.linalg.eigvalsh(A); cas_np.append(time.perf_counter() - z)
        t_moj.append(min(cas_moj)); t_np.append(min(cas_np))
    ni_f = np.array(ni, dtype=float)
    plt.figure(figsize=(4.8, 3.2))
    plt.loglog(ni, t_moj, "o-", label="eigen (naša)")
    plt.loglog(ni, t_np, "s-", label="numpy eigvalsh")
    plt.loglog(ni, ni_f ** 2 * t_moj[0] / ni_f[0] ** 2, "k--", label=r"$\propto n^2$")
    plt.xlabel("n"); plt.ylabel("čas [s]"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{MAPA}/casi.png", dpi=200); plt.close()
    naklon = np.polyfit(np.log(ni_f), np.log(t_moj), 1)[0]
    return ni, t_moj, t_np, naklon


# ============================ številke ============================

class StejocPremik:
    """Ovojnica okoli strategije premika, ki šteje, kolikokrat je bila uporabljena.

    Ima isti vmesnik kot strategija, ki jo ovija (metodo premik), zato jo lahko
    podamo funkciji eigen namesto nje.
    """

    def __init__(self, strategija):
        self.strategija = strategija
        self.klici = 0

    def premik(self, d, e, lo, hi):
        self.klici += 1
        return self.strategija.premik(d, e, lo, hi)


def statistika_premikov(A):
    """Vrne (lastne vrednosti, število korakov, število korakov z varovalom).

    Parametri: A — simetrična matrika.
    Šteje klice qr_korak_premik in _wilkinsonov_premik; zamenjavo razveljavimo
    v bloku finally, zato modul po meritvi ostane nedotaknjen.
    """
    modul = sys.modules["dn1_qr_premik.eigen"]
    izvirni_korak = modul.qr_korak_premik
    izvirno_varovalo = modul._wilkinsonov_premik
    stevec = {"korakov": 0, "varovalo": 0}

    def steti_korak(d, e, mu, V=None):
        stevec["korakov"] += 1
        return izvirni_korak(d, e, mu, V)

    def steti_varovalo(d, e, hi):
        stevec["varovalo"] += 1
        return izvirno_varovalo(d, e, hi)

    modul.qr_korak_premik = steti_korak
    modul._wilkinsonov_premik = steti_varovalo
    try:
        lastne = modul.eigen(A, EnojniPremik())
    finally:
        modul.qr_korak_premik = izvirni_korak
        modul._wilkinsonov_premik = izvirno_varovalo
    return lastne, stevec["korakov"], stevec["varovalo"]


def izpis_natancnost(velikosti=(10, 40, 100, 200)):
    """Tabela natančnosti: odstopanje od oraklja, ortogonalnost in ostanek."""
    print("\n--- Natančnost (naključne simetrične matrike, seme 42) ---")
    print(f"{'n':>5s} {'max|dlambda|':>13s} {'||VtV - I||':>13s} {'||AV - V L||':>13s}")
    rng = np.random.default_rng(42)
    for n in velikosti:
        A = _simetricna(rng, n)
        lam, V = eigen(A, EnojniPremik(), vektorji=True)
        print(f"{n:5d} {np.abs(lam - np.linalg.eigvalsh(A)).max():13.1e}"
              f" {np.abs(V.T @ V - np.eye(n)).max():13.1e}"
              f" {np.abs(A @ V - V @ np.diag(lam)).max():13.1e}")


def izpis_konvergenca(zgod):
    """Zaporedje |e| po korakih in ocena reda konvergence."""
    print("\n--- Konvergenca zadnjega obdiagonalnega elementa (seme 1, n = 12) ---")
    print("  " + " -> ".join(f"{v:.1e}" for v in zgod[:7]))
    if len(zgod) >= 3 and min(zgod[-3:]) > 0:
        e2, e1, e0 = zgod[-3], zgod[-2], zgod[-1]
        red = np.log(e0 / e1) / np.log(e1 / e2)
        print(f"  ocena reda iz zadnjih treh členov: {red:.2f}  (kubično = 3)")


def izpis_korakov(velikosti=(20, 40, 60, 100, 200)):
    """Koliko korakov porabimo na lastno vrednost in kako pogosto se sproži varovalo."""
    print("\n--- Koraki in varovalo ---")
    print(f"{'matrika':>22s} {'korakov':>8s} {'kor./l.v.':>10s} {'varovalo':>9s} {'delež':>7s}")
    primeri = []
    rng = np.random.default_rng(1)
    for n in velikosti:
        primeri.append((f"naključna {n}x{n}", _simetricna(rng, n), n))
    n = 50
    primeri.append(("1D Laplace 50x50",
                    2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1), n))
    primeri.append(("antidiagonalna 2x2", np.array([[0.0, 1.0], [1.0, 0.0]]), 2))

    skupaj = z_varovalom = 0
    for ime, A, n in primeri:
        _, k, w = statistika_premikov(A)
        skupaj += k
        z_varovalom += w
        print(f"{ime:>22s} {k:8d} {k / n:10.2f} {w:9d} {100 * w / max(k, 1):6.1f}%")
    print(f"  skupaj: {z_varovalom} od {skupaj} korakov "
          f"= {100 * z_varovalom / skupaj:.1f} % korakov z varovalom")


def izpis_casi(ni, t_moj, t_np, naklon):
    """Časi izračuna in izmerjeni naklon v log-log merilu."""
    print("\n--- Časi ---")
    print(f"{'n':>5s} {'naša [ms]':>11s} {'numpy [ms]':>11s} {'razmerje':>9s}")
    for n, tm, tn in zip(ni, t_moj, t_np):
        print(f"{n:5d} {tm * 1000:11.1f} {tn * 1000:11.2f} {tm / tn:8.0f}x")
    print(f"  naklon log-log: {naklon:.2f}")


def izpis_grucenje():
    """Lastne vrednosti Laplaceove matrike in uspešnost spektralnega razreza."""
    print("\n--- Spektralno gručenje (seme 2, sigma = 0.8) ---")
    rng = np.random.default_rng(2)
    X = _podatki_gruc(rng)
    L = laplaceova_matrika(podobnostna_matrika(X, sigma=0.8))
    lam, V = eigen(L, EnojniPremik(), vektorji=True)
    f = V[:, 1]
    pravilnih = np.sum((f[:60] > 0) == (f[0] > 0)) + np.sum((f[60:] > 0) != (f[0] > 0))
    print(f"  lambda_1 = {abs(lam[0]):.1e} (numerično nič)")
    print(f"  vrzel lambda_3 / lambda_2 = {lam[2] / lam[1]:.1f}")
    print(f"  Fiedlerjev razrez pravilno razvrsti {pravilnih}/{len(X)} točk")


if __name__ == "__main__":
    print(f"numpy {np.__version__}")
    zgod = slika_konvergenca()
    slika_gruce()
    slika_spekter()
    ni, t_moj, t_np, naklon = slika_casi()
    print(f"slike shranjene v {MAPA}/")

    izpis_natancnost()
    izpis_konvergenca(zgod)
    izpis_korakov()
    izpis_casi(ni, t_moj, t_np, naklon)
    izpis_grucenje()