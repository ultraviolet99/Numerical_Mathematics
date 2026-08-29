"""DN2 (18.2.6): naravni parameter krivulje (t^3 - t, t^2 - 1)."""
from .param import s, hitrost
from .quad import gauss_legendre, integral
from .cheb import cheb_tocke, cheb_koeficienti, clenshaw

__all__ = ["s", "hitrost", "gauss_legendre", "integral",
           "cheb_tocke", "cheb_koeficienti", "clenshaw"]

