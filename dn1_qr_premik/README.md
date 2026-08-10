# DN1 — QR iteracija z enojnim premikom (18.1.9)

**Avtor:** Urban Vesel

Implementacija iskanja lastnih vrednosti in lastnih vektorjev simetrične matrike:
Hessenbergova (tridiagonalna) redukcija, nato QR iteracija s premikom μ = t_nn,
kjer je QR razcep tridiagonalne matrike izveden z Givensovimi rotacijami v O(n)
na iteracijo. Uporaba: spektralna analiza Laplaceove matrike grafa podobnosti.

## Uporaba

```python
import numpy as np
from dn1_qr_premik import eigen, EnojniPremik

A = np.array([[2., 1., 0.], [1., 3., 1.], [0., 1., 4.]])
lastne = eigen(A, EnojniPremik())                      # samo lastne vrednosti
lastne, V = eigen(A, EnojniPremik(), vektorji=True)    # + lastni vektorji (stolpci V)
```

## Testi

```bash
pip install -e ".[dev]"   # namestitev z orodji za razvoj
pytest
pytest --cov=dn1_qr_premik --cov-branch --cov-report=term-missing
```
Pokritost: 100 % stavkov in 100 % vej.

## Demo skripta

Rezultate in slike za poročilo ustvari `report/vizualizacije.py` (poganjaj iz
korena paketa, PNG-ji nastanejo v `report/slike/`):

```bash
python report/vizualizacije.py
```

## Poročilo

```bash
cd report && pdflatex porocilo.tex && pdflatex porocilo.tex
```
