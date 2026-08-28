"""Lastne vrednosti in vektorji simetrične matrike s QR iteracijo z enojnim premikom.

Postopek (18.1.9): matriko najprej s Householderjevimi zrcaljenji prevedemo na
simetrično tridiagonalno T = (d, e), nato ponavljamo korake

mu = d[-1]
T - mu*I = QR
T <- RQ + mu*I

izvedene z Givensovimi rotacijami neposredno na vektorjih (d, e) v O(n) na
korak, z deflacijo, ko obdiagonalni element postane zanemarljiv.
"""
import numpy as np


class EnojniPremik:
    """Strategija premika mu = d[-1] (zadnji diagonalni element aktivnega bloka).
    """
    def premik(self, d, e, lo, hi):
        return d[hi - 1]

def givens(a, b):
    """Givensova rotacija: vrne (c, s, r), da je [[c, s], [-s, c]] @ [a, b] = [r, 0].

    Parametri: a, b — realni števili.
    Vrne: (c, s, r) s c^2 + s^2 = 1 in r = c*a + s*b.
    """
    if b == 0:
        return 1, 0, a
    
    if abs(b) > abs(a):
        t = a / b
        u = np.sign(b) * np.sqrt(1 + t**2)
        s = 1 / u
        c = s * t
        r = b * u
    else:
        t = b / a
        u = np.sign(a) * np.sqrt(1 + t**2)
        c = 1 / u
        s = c * t
        r = a * u

    return c, s, r


def tridiag(A):
    """Householderjeva redukcija simetrične A na tridiagonalno obliko.

    Parametri: A — simetrična n x n (vhod se ne spremeni).
    Vrne: (d, e, U) z A = U T U^T, T = tridiag(d, e), U ortogonalna;
    d dolžine n, e dolžine n-1.
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A ni kvadratna")
    U = np.eye(n)

    for k in range(n - 2):
        x = A[k + 1:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue
        alpha = -np.copysign(norm_x, x[0])
        v = x.copy()
        v[0] -= alpha
        beta = 2 / np.dot(v, v)

        # Posodobitev A
        p = beta * A[k + 1:, k + 1:] @ v
        w = p - (beta * np.dot(p, v) / 2) * v
        A[k + 1:, k + 1:] -= np.outer(v, w) + np.outer(w, v)

        # Zapiši alfa na obdiagonalo in izniči ostanek stolpca
        A[k + 1, k] = alpha
        A[k, k + 1] = alpha
        A[k + 2:, k] = 0
        A[k, k + 2:] = 0

        # Posodobitev U
        z = U[:, k + 1:] @ v
        U[:, k + 1:] -= beta * np.outer(z, v)

    return A.diagonal().copy(), A.diagonal(offset=1).copy(), U


def qr_korak_premik(d, e, mu, V=None):
    """En korak QR iteracije s premikom mu na tridiagonalni T = (d, e).

    Izvede T <- R Q + mu*I za razcep T - mu*I = Q R in vrne nova (d, e);
    če je V podana (n x m), jo hkrati posodobi V <- V Q (na mestu).

    Parametri: d (m,), e (m-1,), mu — premik, V — opcijska matrika vektorjev.
    Vrne: (d, e) po koraku. Rezultat je spet simetrična tridiagonalna
    """
    m = len(d)
    if m == 1:
        return d, e
    
    a = d - mu
    c = np.empty(m - 1)
    s = np.empty(m - 1)
    r0 = np.empty(m)
    r1 = np.empty(m - 1)

    p = a[0]
    q = e[0]
    for i in range(m - 1):
        c[i], s[i], r0[i] = givens(p, e[i])
        r1[i] = c[i] * q + s[i] * a[i + 1]
        p = -s[i] * q + c[i] * a[i + 1]
        if i < m - 2:
            q = c[i] * e[i + 1]
        if V is not None:
            V_i, V_ip1 = V[:, i].copy(), V[:, i + 1].copy()
            V[:, i] = c[i] * V_i + s[i] * V_ip1
            V[:, i + 1] = -s[i] * V_i + c[i] * V_ip1
    r0[m - 1] = p

    c_prej = 1

    for i in range(m - 1):
        d[i] = c_prej * c[i] * r0[i] + s[i] * r1[i] + mu
        e[i] = s[i] * r0[i + 1]
        c_prej = c[i]
    d[m - 1] = c_prej * r0[m - 1] + mu

    return d, e

def _wilkinsonov_premik(d, e, hi):
    """Wilkinsonov premik za spodnji 2x2 blok — varovalo ob stagnaciji.

    Vrne lastno vrednost bloka [[d[hi-2], e[hi-2]], [e[hi-2], d[hi-1]]]
    """
    delta = (d[hi - 2] - d[hi - 1]) / 2
    b = e[hi - 2]
    sign_delta = np.sign(delta) if delta != 0 else 1
    mu = d[hi - 1] - sign_delta * b**2 / (abs(delta) + np.sqrt(delta**2 + b**2))
    return mu

class WilkinsonovPremik:
    """Strategija premika mu = Wilkinsonov premik spodnjega 2x2 bloka.

    Uporablja se kot varovalo ob stagnaciji.
    """
    def premik(self, d, e, lo, hi):
        return _wilkinsonov_premik(d, e, hi)

_VAROVALO = WilkinsonovPremik()


def eigen(A, metoda=None, vektorji=False, tol=None, max_iter=10_000):
    """Vse lastne vrednosti (in vektorji) simetrične matrike A.

    Klic zrcali specifikacijo naloge:
        lastne = eigen(A, EnojniPremik())
        lastne, V = eigen(A, EnojniPremik(), vektorji=True)

    Parametri: A - simetrična n x n; metoda - strategija premika (EnojniPremik);
    vektorji - ali vrnemo tudi V; tol - deflacijski prag (privzeto strojni eps);
    max_iter - varovalo skupnega števila korakov.
    Vrne: naraščajoče urejene lastne vrednosti; z vektorji=True še ortogonalno
    V (stolpci so lastni vektorji, A V ~ V diag(lastne)).
    Napake: ValueError za nesimetričen/nekvadraten vhod, RuntimeError ob
    prekoračitvi max_iter.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A ni kvadratna")
    if not np.allclose(A, A.T, atol=1e-8 * (1 + np.abs(A).max())):
        raise ValueError("A mora biti simetrična")
    if tol is None:
        tol = np.finfo(float).eps
    if n == 1:
        lastne = np.array([A[0, 0]])
        V = np.eye(1) if vektorji else None
        return (lastne, V) if vektorji else lastne
    if metoda is None:
        metoda = EnojniPremik()
    if not hasattr(metoda, "premik"):
        raise ValueError("metoda mora imet premik(d, e, lo, hi)")

    d, e, U = tridiag(A)
    V = U if vektorji else None

    hi = n
    stagnacija = 0
    prejsnji_rep = np.inf
    prejsnji_hi = -1
    for _ in range(max_iter):
        # Deflacija
        for i in range(hi - 1):
            if abs(e[i]) <= tol * (abs(d[i]) + abs(d[i + 1])):
                e[i] = 0
        while hi > 1 and e[hi - 2] == 0:
            hi -= 1
        if hi <= 1:
            break

        if hi != prejsnji_hi:      # blok se je spremenil
            stagnacija = 0
            prejsnji_rep = np.inf
            prejsnji_hi = hi
        rep = abs(e[hi - 1 - 1])
        stagnacija = stagnacija + 1 if rep > 0.5 * prejsnji_rep else 0
        prejsnji_rep = rep

        lo = hi - 1
        while lo > 0 and e[lo - 1] != 0:
            lo -= 1

        izbrana_met = _VAROVALO if (stagnacija >= 3 and hi - lo >= 2) else metoda

        # QR korak s premikom na bloku d[lo:hi], e[lo:hi-1]
        mu = izbrana_met.premik(d, e, lo, hi)
        qr_korak_premik(d[lo:hi], e[lo:hi - 1], mu, V[:, lo:hi] if vektorji else None)

    else:
        raise RuntimeError("Prekoračeno število iteracij")

    # Sortiranje lastnih vrednosti in permutacija stolpcev V
    idx = np.argsort(d)
    d_sorted = d[idx]
    if vektorji:
        V_sorted = V[:, idx]
        return d_sorted, V_sorted
    return d_sorted