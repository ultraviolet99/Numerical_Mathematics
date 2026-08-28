"""Graf podobnosti in njegova Laplaceova matrika (pogl. 8), za zahtevani preizkus.

Spektralno gručenje: najmanjša lastna vrednost L je 0 (konstanten vektor),
predznak druge (Fiedlerjeve) lastne komponente pa razreže točke v gruči.
"""
import numpy as np


def podobnostna_matrika(X, sigma=1.0):
    """Utežena matrika podobnosti W točk X z Gaussovim jedrom.

    Parametri: X - n x d matrika točk (vsaka vrstica ena točka); sigma > 0.
    Vrne: simetrično W (n x n) z W[i, j] = exp(-||x_i - x_j||^2/(2 sigma^2))
    za i != j in W[i, i] = 0
    """
    k = np.sum(X**2, axis=1)
    D2 = k[:, None] + k[None, :] - 2 * X @ X.T
    D2 = np.maximum(D2, 0)
    W = np.exp(-D2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)
    return W


def laplaceova_matrika(W):
    """Laplaceova matrika grafa: L = D - W, D = diag(vsote vrstic W).

    Parametri: W - simetrična utežena matrika sosednosti (n x n).
    Vrne: simetrično pozitivno semidefinitno L (n x n); L @ 1 = 0.
    """
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L
