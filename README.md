# Scenario analysis of a delta-hedged spread option (MF731 HW 9)

Long one option paying (S1_T - S2_T)^2, delta-hedged, 5-day horizon, 3888 scenarios.

- (a) V = e^{r tau}(s1^2 e^{sigma1^2 tau} - 2 s1 s2 e^{rho sigma1 sigma2 tau} + s2^2 e^{sigma2^2 tau}).
  V0 = 921.74 (MC 921.44 +- 0.71), h1 = -21.95, h2 = 31.20, hedged book Vt = -V0.
- (b) worst loss 1,000.36 at X1 = X2 = +0.20, both vol ratios 0.5, rho + 0.5.
- (c) max w l = 1,313.79 at X1 = X2 = +0.05, both vol ratios 0.75, rho + 0.5;
  weight 1.953125, loss 672.66.

```bash
pip install -r requirements.txt
python src/scenario.py     # writes generated/*.tex and figures/heatmap.png
latexmk -pdf main.tex
```
