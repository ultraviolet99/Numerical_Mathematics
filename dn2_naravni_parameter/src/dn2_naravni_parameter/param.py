"""Naravni parameter s(t) = int_0^t sqrt(x'(tau)^2 + y'(tau)^2) dtau
za krivuljo (x, y) = (t^3 - t, t^2 - 1).

Iz x' = 3 tau^2 - 1, y' = 2 tau sledi integrand g = sqrt(9 tau^4 - 2 tau^2 + 1);
polinom pod korenom je povsod pozitiven (minimum 8/9 pri tau^2 = 1/9), zato je
g na realni osi analitičen, s pa liha, gladka, strogo naraščajoča.

Zahtevi: relativna napaka < 5e-11 za vse argumente
"""
from functools import lru_cache

import numpy as np

from .cheb import cheb_koeficienti, cheb_tocke, clenshaw
from .quad import integral

T0 = 3.0     # meja med vejama
N_GL = 60    # red kvadrature, utemeljitev in konvergenčna slika v poročilu
M_CHEB = 40  # stopnja interpolacije h, utemeljitev in slika padanja |c_k| v poročilu


def hitrost(tau):
    """Dolžina hitrostnega vektorja |r'(tau)| = sqrt(9 tau^4 - 2 tau^2 + 1).

    Parametri: tau - število ali numpy tabela (vektorizirano).
    Vrne: vrednost(i) >= sqrt(8/9); soda funkcija.
    """
    tau = np.asarray(tau, dtype=float)
    t2 = tau * tau
    return np.sqrt((9.0 * t2 - 2.0) * t2 + 1.0)


@lru_cache(maxsize=1)
def _h_koeficienti():
    """Čebiševi koeficienti funkcije h(u) = u^3 s(1/u) na [0, 1/T0].

    Vrne: nespremenljivo tabelo koeficientov (dolžina M_CHEB + 1); izračuna se
    natanko enkrat (lru_cache) - od tod O(1) cena interpolacijske veje.
    """
    u = cheb_tocke(M_CHEB, 0, 1 / T0)
    h = np.empty(M_CHEB + 1)
    for j in range(M_CHEB + 1):
        if u[j] == 0.0:
            h[j] = 1.0
        else:
            h[j] = u[j] ** 3 * (
                integral(hitrost, 0, T0, N_GL)
                + integral(hitrost, T0, 1 / u[j], N_GL)
            )
    return cheb_koeficienti(h)


def s(t):
    """Naravni parameter (ločna dolžina) krivulje (t^3 - t, t^2 - 1) od 0 do t.

    Parametri: t - realno število (int ali float).
    Vrne: float; relativna napaka < 5e-11 za vse t. 
    Cena klica O(1) (po enkratni pripravi). 
    Robna primera: NaN -> NaN; za |t| ~> 5.6e102 vrednost
    t^3 preseže obseg float64 in rezultat je pa korektno inf.
    """
    if t != t:  # NaN
        return t
    if t < 0:
        return -s(-t)
    if t == 0:
        return 0.0
    if t <= T0:
        return integral(hitrost, 0, t, N_GL)
    c = _h_koeficienti()
    u = 1 / t
    xi = 2 * T0 * u - 1
    return t * t * t * clenshaw(c, xi)
