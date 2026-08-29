# DN3 - Ničle Airyjeve funkcije (18.3.1)

**Avtor:** Urban Vesel

Iskanje ničel funkcije Ai na 10 decimalk: enačbo Ai'' = x*Ai zapišemo kot sistem
y' = A(x)*y in ga integriramo z lastno implementacijo Magnusove metode reda 4
(matrična eksponentna funkcija 2x2 v zaprti obliki). Ničle lociramo prek menjav
predznaka in prečistimo z varovano Newtonovo metodo (Ai' integriramo zraven).
`scipy.special` je uporabljen izključno v testih za preverjanje (analog `airyai`).

## Uporaba

```python
from dn3_airy_nicle import nicle
nicle(5)          # prvih 5 nicel: [-2.3381..., -4.0879..., ...]
nicle(x_min=-30)  # vse nicle na intervalu [-30, 0]
```

## Testi

```bash
pip install -e ".[dev]"   # namestitev z orodji za razvoj
pytest
pytest --cov=dn3_airy_nicle --cov-branch --cov-report=term-missing
```
Pokritost: 100 % (stavki in veje).

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
