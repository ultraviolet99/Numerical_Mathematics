# DN2 - Naravni parameter (18.2.6, težja)

**Avtor:** Urban Vesel

Izračun naravnega parametra (ločne dolžine) s(t) parametrične krivulje
(x, y) = (t³ - t, t² - 1) z relativno natančnostjo 5e-11 za vse argumente in
s časovno zahtevnostjo, omejeno s konstanto: Gauss-Legendrove kvadrature za
zmerne t in Čebiševa interpolacija funkcije u -> u³ s(1/u) za velike t.

## Uporaba

```python
from dn2_naravni_parameter import s
s(1.0)      # locna dolzina od t=0 do t=1
s(1e8)      # deluje enako hitro tudi za velike argumente
```

## Testi

```bash
pip install -e ".[dev]"   # namestitev z orodji za razvoj
pytest
pytest --cov=dn2_naravni_parameter --cov-branch --cov-report=term-missing
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

