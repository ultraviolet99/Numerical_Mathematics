"""Gauss-Legendrova kvadratura (lastna implementacija, brez scipy).

Vozli so ničle Legendrovega polinoma P_n, uteži w_i = 2/((1 - x_i^2) P_n'(x_i)^2).
Za integrande, analitične v okolici intervala, napaka pada kot rho^(-2n)
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def gauss_legendre(n):
    """Vozli in uteži Gauss-Legendrove kvadrature reda n na [-1, 1].

    Parametri: n - število vozlov (>= 1).
    Vrne: (x, w) - tabeli dolžine n. x naraščajoče simetrični okoli 0,
    w > 0 in vsota w = 2. Rezultat se shrani (lru_cache), zato se vsak
    red izračuna natanko enkrat. To zagotavlja O(1) ceno klica s(t).

    """
    x = np.zeros(n)
    w = np.zeros(n)
    for i in range(1, n + 1):
        x[i - 1] = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        for _ in range(6):  # Newton, kvadratična konvergenca, 5 korakov dovolj do n = 500
            p0, p1 = 1.0, x[i - 1]
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x[i - 1] * p1 - (k - 1) * p0) / k
            dp = n * (x[i - 1] * p1 - p0) / (x[i - 1] * x[i - 1] - 1)
            dx = -p1 / dp
            x[i - 1] += dx
        w[i - 1] = 2 / ((1 - x[i - 1] * x[i - 1]) * dp * dp)
    x.setflags(write=False); w.setflags(write=False)
    return -x, w   # x pada od skoraj 1 do skoraj -1, torej -x narašča

def integral(f, a, b, n=60):
    """Približek integrala f na [a, b] z GL kvadraturo reda n.

    Parametri: f - funkcija, ki sprejme numpy tabelo (vektorizirana).
    a, b - meji; n - red kvadrature.
    Vrne: float. Točno za polinome stopnje <= 2n - 1.
    """
    x, w = gauss_legendre(n)
    x = np.array(x)
    w = np.array(w)
    t = (a + b) / 2 + (b - a) / 2 * x
    return (b - a) / 2 * np.sum(w * f(t))
