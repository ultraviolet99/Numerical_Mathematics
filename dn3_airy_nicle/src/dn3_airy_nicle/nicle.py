"""Iskanje ničel Airyjeve funkcije Ai na 10 decimalk (18.3.1).

Začetna pogoja (18.35; DLMF 9.2.3-9.2.4):
    Ai(0)  = 1/(3^(2/3) Gamma(2/3)),   Ai'(0) = -1/(3^(1/3) Gamma(1/3));
ker Gamma ni elementarna, sta konstanti zapisani na 20 mest (float64 ju
pravilno zaokroži) in dokumentirani z zaprto obliko.
"""
import numpy as np

from .magnus import integriraj, korak_dolzina, magnus_korak

AI0 = 0.35502805388781723926    # Ai(0)  = 1/(3^(2/3) Gamma(2/3)), DLMF 9.2.3
DAI0 = -0.25881940379280679840  # Ai'(0) = -1/(3^(1/3) Gamma(1/3)), DLMF 9.2.4

_CACHE = {"xs": None, "ys": None}  # predpomnjena mreža (xs, ys) za ai(x)


def _mreza(x_min=-30.0):
    """Vrni predpomnjeno mrežo (xs, ys) za ai(x).

    Parametri: x_min -- globina mreže (x_min < 0).
    Vrne: (xs, ys) -- padajoče urejena numpy tabela vozlov.
    """
    if _CACHE["xs"] is None:
        _CACHE["xs"], _CACHE["ys"] = integriraj(np.array([AI0, DAI0]),
                                                 x_min=min(-30.0, x_min))
    elif _CACHE["xs"][-1] > x_min:                    # zahtevaš globlje, kot imaš
        xs2, ys2 = integriraj(_CACHE["ys"][-1], x0=_CACHE["xs"][-1],
                              x_min=x_min - 5.0)       # rezerva, da ne podaljšuješ ob vsakem klicu
        _CACHE["xs"] = np.concatenate([_CACHE["xs"], xs2[1:]])
        _CACHE["ys"] = np.concatenate([_CACHE["ys"], ys2[1:]])
    return _CACHE["xs"], _CACHE["ys"]


def ai(x, **kw):
    """Vrednosti (Ai(x), Ai'(x)) za x <= 0 s propagacijo od predpomnjene mreže.

    Parametri: x -- točka (x <= 0); kw -- opcijska parametra h0 in
    korakov_na_periodo za korak_dolzina.
    Vrne: (Ai, Ai') kot par float vrednosti.

    Pomožna funkcija za Newtona in za grafe; ni splošna implementacija Ai.
    """
    x = float(x)
    if x > 0.0:
        raise ValueError("x > 0 ni podprto")
    xs, ys = _mreza(x)
    i = np.searchsorted(xs[::-1], x, side="left")
    j = len(xs) - 1 - i
    t = xs[j]
    y = ys[j].copy()
    while t > x + 1e-14:
        h = min(korak_dolzina(t, **kw), t - x)
        y = magnus_korak(t, y, -h)
        t -= h
    return float(y[0]), float(y[1])

def _precisti(xL, xR, tol=1e-11, max_iter=15, f=None):
    """Prečisti ničlo na oklepu [xL, xR] z varovano Newtonovo metodo.

    Parametri: xL, xR -- krajišči oklepa, na katerih ima f nasprotna predznaka;
    tol -- ustavitveni kriterij (relativno na 1 + |x|) 
    max_iter -- zgornja meja iteracij
    f -- funkcija, ki vrne (vrednost, odvod), privzeto ai.
    Vrne: približek ničle v [xL, xR].

    V vsaki iteraciji najprej zožimo oklep glede na predznak, nato naredimo
    Newtonov korak. Če ta pade iz oklepa, ga nadomestimo z razpoloviščem, tako
    da se postopek v najslabšem primeru izrodi v bisekcijo in vseeno konvergira.
    """
    if f is None:
        f = ai
    vR = f(xR)[0]
    x = (xL + xR) / 2.0
    for _ in range(max_iter):
        v, dv = f(x)
        if v * vR > 0.0:                 # oži oklep glede na predznak
            xR, vR = x, v
        else:
            xL = x
        x_new = x - v / dv               # Newtonov korak
        # varovalo: če x_new pade iz oklepa, uporabi bisekcijo (mejo dovolimo,
        # ker Newton ob skoraj konvergenci lahko pristane točno na njej)
        if not (xL <= x_new <= xR):
            x_new = (xL + xR) / 2.0
        if abs(x_new - x) <= tol * (1.0 + abs(x)) / 10.0:
            return x_new
        x = x_new
    return x

def nicle(n=None, x_min=None, tol=1e-11):
    """Ničle funkcije Ai: prvih n ali vse na [x_min, 0].

    Parametri: podaj vsaj enega od n (število ničel) / x_min (globina);
    tol -- ustavitveni kriterij Newtona (relativno na 1 + |x|).
    Vrne: padajoče urejeno numpy tabelo a_1 > a_2 > ... (vse < 0), vsaka na
    10 decimalnih mest natančno.
    """
    if n is None and x_min is None:
        raise ValueError("podaj vsaj n ali x_min")
    if n is not None and n <= 0:
        raise ValueError("n mora biti > 0")
    if x_min is not None and x_min >= 0.0:
        raise ValueError("x_min mora biti < 0")

    # oceni globino, če je podan le n
    if x_min is None:
        x_min = -((3.0 * np.pi * n) / 2.0) ** (2.0 / 3.0) * 1.1

    # pohod od 0 do x_min z detekcijo menjav predznaka Ai
    xs, ys = integriraj(np.array([AI0, DAI0]), x0=0.0, x_min=x_min)
    oklepi = []
    for i in range(len(xs) - 1):
        if ys[i][0] * ys[i + 1][0] < 0.0:
            oklepi.append((xs[i + 1], xs[i]))  # xs pada, torej xL < xR šele obrnjeno

    # na vsakem oklepu varovan Newton
    a = [_precisti(xL, xR, tol) for (xL, xR) in oklepi]
    a = np.array(a)
    a.sort()
    a = a[::-1]  # padajoče
    if n is not None:
        a = a[:n]
    return a
