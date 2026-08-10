"""Merjenje, kako pogosto se sproži Wilkinsonovo varovalo.
 
Ta različica NE zahteva nobene spremembe funkcije eigen — šteje tako, da za čas
meritve zamenja dve funkciji v modulu in ju na koncu vrne nazaj. Deluje ne
glede na to, ali si popravke 1 in 3 že vnesel.
 
Dodaj v report/vizualizacije.py in pokliči izpisi_statistiko() iz __main__;
številko vpiši v poročilo namesto [DOPOLNI: delež].
"""
import sys
 
import numpy as np
 
from dn1_qr_premik import EnojniPremik
 
 
def statistika_premikov(A):
    """Vrne (lastne vrednosti, število korakov, število korakov z varovalom).
 
    Parametri: A — simetrična matrika.
    Šteje klice funkcij qr_korak_premik in _wilkinsonov_premik. Zamenjavo
    razveljavimo v bloku finally, zato modul po meritvi ostane nedotaknjen.
    """
    # Pozor: v paketu ime "eigen" kaže na funkcijo, ne na modul (tako je
    # nastavljen __init__.py), zato modul poiščemo v sys.modules.
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
 
 
def izpisi_statistiko():
    """Izpiše preglednico za poročilo: delež korakov, v katerih se sproži varovalo."""
    primeri = []
    rng = np.random.default_rng(1)
    for n in (20, 40, 60, 100, 200):
        B = rng.standard_normal((n, n))
        primeri.append((f"naključna {n}x{n}", (B + B.T) / 2))
    n = 50
    primeri.append(("1D Laplace 50x50",
                    2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)))
    primeri.append(("antidiagonalna 2x2", np.array([[0.0, 1.0], [1.0, 0.0]])))
 
    print(f"{'matrika':>22s} {'korakov':>8s} {'varovalo':>9s} {'delež':>7s}")
    skupaj = z_varovalom = 0
    for ime, A in primeri:
        _, k, w = statistika_premikov(A)
        skupaj += k
        z_varovalom += w
        print(f"{ime:>22s} {k:8d} {w:9d} {100 * w / max(k, 1):6.1f}%")
    print(f"\nskupaj: {z_varovalom} od {skupaj} korakov "
          f"= {100 * z_varovalom / skupaj:.1f} %")
 
 
if __name__ == "__main__":
    izpisi_statistiko()
