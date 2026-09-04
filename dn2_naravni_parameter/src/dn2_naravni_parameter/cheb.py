"""Interpolacija v Čebiševih točkah
"""
import numpy as np


def cheb_tocke(m, a=-1.0, b=1.0):
    """m + 1 Čebiševih točk druge vrste na [a, b].

    Parametri: m - stopnja (točk je m + 1); a, b - interval.
    Vrne: tabelo x_j = (a+b)/2 + (b-a)/2 * cos(pi j/m), j = 0..m,
    Padajoče od b proti a
    """
    j = np.arange(m + 1)
    return (a + b) / 2 + (b - a) / 2 * np.cos(np.pi * j / m)


def cheb_koeficienti(f_vrednosti):
    """Koeficienti c_0 ... c_m interpolanta sum_k c_k T_k iz vrednosti funkcije.

    Parametri: f_vrednosti - tabela dolžine m + 1 vrednosti funkcije v točkah
    cheb_tocke(m, a, b) (isti vrstni red).
    Vrne: tabelo c dolžine m + 1 za neposredno rabo v clenshaw.
    """
    m = len(f_vrednosti) - 1
    c = np.zeros(m + 1)
    for k in range(m + 1):
        c[k] = f_vrednosti[0] / 2 + sum(
            f_vrednosti[j] * np.cos(np.pi * j * k / m) for j in range(1, m)
        ) + (-1) ** k * f_vrednosti[m] / 2
        c[k] *= 2 / m
    c[0] /= 2
    c[m] /= 2
    return c

def clenshaw(c, xi):
    """Vrednost sum_k c_k T_k(xi) za xi iz [-1, 1] s Clenshawovo rekurzijo.

    Parametri: c - koeficienti (dolžina m + 1). xi - točka v [-1, 1].
    Vrne: float. Povratno stabilno, O(m), brez eksplicitnih T_k.
    """
    m = len(c) - 1
    b1 = b2 = 0
    for k in range(m, 0, -1):
        b1, b2 = c[k] + 2 * xi * b1 - b2, b1
    return c[0] + xi * b1 - b2
