"""Magnusova metoda reda 4 za y' = A(x) y (enačbi 18.36-18.37 iz skripte).

En korak z x na x + h:
    A1 = A(x + (1/2 - sqrt(3)/6) h),   A2 = A(x + (1/2 + sqrt(3)/6) h),
    sigma = (h/2)(A1 + A2) - (sqrt(3)/12) h^2 [A1, A2],
    y_nov = exp(sigma) y,                       [A, B] = AB - BA,
z lokalno napako O(h^5). Za Airyjev sistem (A(x) = [[0,1],[x,0]], brezsledna)
se komutator sklene: [A1, A2] = (x2 - x1) diag(1, -1), x2 - x1 = (sqrt(3)/3) h,
zato ima sigma zaprto obliko

    sigma = [[-h^3/12, h], [h (x + h/2), h^3/12]].

Ker je tr sigma = 0, je det(exp sigma) = 1 eksaktno. Metoda zato strukturno
ohranja Wronskijan (Abel-Liouville), enako kot točen tok.
"""
import numpy as np

SQRT3 = np.sqrt(3.0)


def A_airy(x):
    """Matrika Airyjevega sistema A(x) = [[0, 1], [x, 0]] za y = (Ai, Ai').

    Parametri: x - realno število. Vrne: 2 x 2 numpy matriko (brezsledno).
    """
    return np.array([[0.0, 1.0], [x, 0.0]])


def expm2(sigma):
    """exp(sigma) za brezsledno 2 x 2 matriko v zaprti obliki.

    Parametri: sigma - 2 x 2 z tr sigma = 0.
    Vrne: 2 x 2 matriko exp(sigma) = C(z) I + S(z) sigma, kjer je
    z = sigma[0,0]^2 + sigma[0,1] sigma[1,0] (= -det sigma) in
    C(z) = cosh(sqrt z), S(z) = sinh(sqrt z)/sqrt z - CELI funkciji z; 
    det rezultata je 1 do zaokrožitev.
    """
    z = sigma[0, 0] ** 2 + sigma[0, 1] * sigma[1, 0]
    delta = 1e-10
    if z > delta:
        w = np.sqrt(z)
        C = np.cosh(w)
        S = np.sinh(w) / w
    elif z < -delta:
        w = np.sqrt(-z)
        C = np.cos(w)
        S = np.sin(w) / w
    else:
        C = 1.0 + z / 2.0 + z ** 2 / 24.0 + z ** 3 / 720.0
        S = 1.0 + z / 6.0 + z ** 2 / 120.0 + z ** 3 / 5040.0

    return C * np.eye(2) + S * sigma


def magnus_korak(x, y, h):
    """En Magnusov korak reda 4 za Airyjev sistem: z x na x + h (h je lahko < 0).

    Parametri: x - trenutna točka; y - stanje (Ai, Ai'), tabela dolžine 2;
    h - dolžina koraka s predznakom.
    Vrne: novo stanje y v točki x + h (tabela dolžine 2).
    """
    sigma = np.array([[-h ** 3 / 12.0, h], [h * (x + h / 2.0), h ** 3 / 12.0]])
    return expm2(sigma) @ y

def korak_dolzina(x, h0=0.05, korakov_na_periodo=800.0):
    """Deterministična dolžina koraka h = min(h0, 2 pi/(m sqrt(1 + |x|))) > 0.

    Parametri: x - trenutna točka; h0 - zgornja meja koraka;
    korakov_na_periodo (m) - ciljno število korakov na lokalni nihaj.
    Vrne: pozitivni float.
    """
    return min(h0, 2.0 * np.pi / (korakov_na_periodo * np.sqrt(1.0 + abs(x))))


def integriraj(y0, x0=0.0, x_min=-30.0, h0=0.05, korakov_na_periodo=800.0):
    """Pohod sistema od x0 navzdol do x_min z Magnusovimi koraki.

    Parametri: y0 -- začetno stanje (dolžina 2); x0, x_min -- meji (x_min < x0);
    h0, korakov_na_periodo -- parametra za korak_dolzina.
    Vrne: (xs, ys) -- vozle (1D, padajoče, xs[0] = x0, xs[-1] = x_min) in
    stanja (len(xs) x 2). Osnova za lociranje ničel in graf Ai.
    """
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    while x > x_min + 1e-14:
        h = min(korak_dolzina(x, h0, korakov_na_periodo), x - x_min)
        y = magnus_korak(x, y, -h)
        x -= h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
