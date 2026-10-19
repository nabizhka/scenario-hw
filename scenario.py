"""Scenario analysis for a delta-hedged long option on (S1 - S2)^2 (MF731 HW 9).

(a) closed-form price checked against Monte Carlo; deltas checked by finite differences.
(b) worst-case loss over the 3888-scenario grid.
(c) weighted scenario risk measure max_n w_n l_n, its scenario, and that scenario's loss.
Writes generated/values.tex, generated/top_b.tex, generated/top_c.tex, figures/heatmap.png.
"""
import itertools
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
rng = np.random.default_rng(731)

# parameters from the sheet
r, mu1, mu2 = 0.0132, 0.15475, 0.18312          # mu1, mu2 never enter: see write-up
sig1, sig2, rho = 0.2214, 0.3036, 0.237
T, S1, S2, D = 0.25, 158.12, 170.33, 1 / 252
H = 5 * D                                        # 5-day horizon


def price(tau, s1, s2, s_1, s_2, c):
    """V(t, s1, s2; sigma, rho) with tau = T - t, part (a)."""
    return np.exp(r * tau) * (s1**2 * np.exp(s_1**2 * tau)
                              - 2 * s1 * s2 * np.exp(s_1 * s_2 * c * tau)
                              + s2**2 * np.exp(s_2**2 * tau))


def deltas(tau, s1, s2, s_1, s_2, c):
    g = np.exp(r * tau)
    h1 = 2 * g * (s1 * np.exp(s_1**2 * tau) - s2 * np.exp(s_1 * s_2 * c * tau))
    h2 = 2 * g * (s2 * np.exp(s_2**2 * tau) - s1 * np.exp(s_1 * s_2 * c * tau))
    return h1, h2


# ---------------------------------------------------------------- (a) checks
n = 4_000_000
z1 = rng.standard_normal(n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
ST1 = S1 * np.exp((r - sig1**2 / 2) * T + sig1 * np.sqrt(T) * z1)
ST2 = S2 * np.exp((r - sig2**2 / 2) * T + sig2 * np.sqrt(T) * z2)
pay = np.exp(-r * T) * (ST1 - ST2) ** 2
V0 = price(T, S1, S2, sig1, sig2, rho)
mc, mc_se = pay.mean(), pay.std(ddof=1) / np.sqrt(n)
assert abs(mc - V0) < 4 * mc_se, (mc, V0, mc_se)

h1, h2 = deltas(T, S1, S2, sig1, sig2, rho)
eps = 1e-4
fd1 = (price(T, S1 + eps, S2, sig1, sig2, rho) - price(T, S1 - eps, S2, sig1, sig2, rho)) / (2 * eps)
fd2 = (price(T, S1, S2 + eps, sig1, sig2, rho) - price(T, S1, S2 - eps, sig1, sig2, rho)) / (2 * eps)
assert abs(fd1 - h1) < 1e-6 and abs(fd2 - h2) < 1e-6
# Euler homogeneity: V is homogeneous of degree 2 in (s1, s2), so s1 h1 + s2 h2 = 2V
assert abs(S1 * h1 + S2 * h2 - 2 * V0) < 1e-8
Vt = V0 - h1 * S1 - h2 * S2                      # = -V0 by homogeneity
assert abs(Vt + V0) < 1e-8

# ---------------------------------------------------------------- scenario grid
X_vals = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
k_vals = [0.5, 0.75, 1.25, 1.5, 1.75, 2.0]       # sigma_{t+5D} / sigma_t
d_vals = [-0.5, 0.0, 0.5]                         # rho_{t+5D} - rho_t
wX = {0.20: 0.5, 0.10: 0.75, 0.05: 1.0}
wk = {0.5: 0.5, 2.0: 0.5, 0.75: 1.25, 1.75: 1.25, 1.25: 0.75, 1.5: 0.75}
wd = {0.0: 0.5, 0.5: 1.25, -0.5: 1.25}

grid = np.array(list(itertools.product(X_vals, X_vals, k_vals, k_vals, d_vals)))
assert len(grid) == 3888
x1, x2, k1, k2, dr = grid.T
w = np.array([wX[abs(a)] * wX[abs(b)] * wk[c] * wk[d] * wd[e] for a, b, c, d, e in grid])
assert wX[0.10] * wX[0.20] * wk[2.0] * wk[0.5] * wd[-0.5] == 15 / 128   # sheet's example

tau1 = T - H
V1 = price(tau1, S1 * np.exp(x1), S2 * np.exp(x2), k1 * sig1, k2 * sig2, rho + dr) \
    - h1 * S1 * np.exp(x1) - h2 * S2 * np.exp(x2)
loss = -(V1 - Vt)

# sanity: no move in prices, vols or correlation -> loss is just 5 days of time decay
V_flat = price(tau1, S1, S2, sig1, sig2, rho) - h1 * S1 - h2 * S2
theta_loss = -(V_flat - Vt)


def describe(i):
    return dict(X1=x1[i], X2=x2[i], k1=k1[i], k2=k2[i], dr=dr[i], rho=rho + dr[i],
                loss=loss[i], w=w[i], wl=w[i] * loss[i])


ib = int(np.argmax(loss))
ic = int(np.argmax(w * loss))
ties_b = int(np.sum(np.isclose(loss, loss[ib], rtol=0, atol=1e-9)))
ties_c = int(np.sum(np.isclose(w * loss, (w * loss)[ic], rtol=0, atol=1e-9)))
B, C = describe(ib), describe(ic)

# decompose the worst loss: delta-hedged price move vs vol/corr repricing
V_px_only = price(tau1, S1 * np.exp(B["X1"]), S2 * np.exp(B["X2"]), sig1, sig2, rho) \
    - h1 * S1 * np.exp(B["X1"]) - h2 * S2 * np.exp(B["X2"])
loss_px_only = -(V_px_only - Vt)

for lab, d in [("(b)", B), ("(c)", C)]:
    print(lab, {k: round(float(v), 4) for k, v in d.items()})
print("V0", V0, "MC", mc, "+-", mc_se, "h", h1, h2, "Vt", Vt, "theta loss", theta_loss)
print("ties", ties_b, ties_c, "loss px only at (b) moves", loss_px_only)
print("positive-loss scenarios", int((loss > 0).sum()), "of 3888; min loss", loss.min())

# ---------------------------------------------------------------- outputs
os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def m(name, val, fmt):
    return f"\\newcommand{{\\{name}}}{{{format(val, fmt)}}}\n"


with open(os.path.join(ROOT, "generated", "values.tex"), "w") as f:
    f.write(m("VzeroV", V0, ",.4f") + m("MCV", mc, ",.4f") + m("MCse", mc_se, ".4f"))
    f.write(m("hOne", h1, ".4f") + m("hTwo", h2, ".4f") + m("Vport", Vt, ",.4f"))
    f.write(m("FDerr", max(abs(fd1 - h1), abs(fd2 - h2)), ".1e"))
    f.write(m("ThetaLoss", theta_loss, ".4f") + m("NposLoss", int((loss > 0).sum()), "d"))
    f.write(m("MinLoss", loss.min(), ",.2f") + m("PxOnlyLoss", loss_px_only, ",.2f"))
    f.write(m("VolPart", B["loss"] - loss_px_only, ",.2f") + m("Wmax", w.max(), ".6f"))
    for tag, d in [("B", B), ("C", C)]:
        f.write(m(f"{tag}Xone", d["X1"], "+.2f") + m(f"{tag}Xtwo", d["X2"], "+.2f"))
        f.write(m(f"{tag}kone", d["k1"], ".2f") + m(f"{tag}ktwo", d["k2"], ".2f"))
        f.write(m(f"{tag}dr", d["dr"], "+.1f") + m(f"{tag}rho", d["rho"], ".3f"))
        f.write(m(f"{tag}sone", d["k1"] * sig1 * 100, ".2f") + m(f"{tag}stwo", d["k2"] * sig2 * 100, ".2f"))
        f.write(m(f"{tag}loss", d["loss"], ",.2f") + m(f"{tag}w", d["w"], ".6f"))
        f.write(m(f"{tag}wl", d["wl"], ",.2f"))
    f.write(m("TiesB", ties_b, "d") + m("TiesC", ties_c, "d"))


def top_table(score, fname, k=10):
    idx = np.argsort(-score)[:k]
    with open(os.path.join(ROOT, "generated", fname), "w") as f:
        f.write("\\begin{tabular}{rrrrrrrrr}\n\\toprule\n")
        f.write("Rank & $X^{(1)}$ & $X^{(2)}$ & $\\sigma^{(1)}$ ratio & $\\sigma^{(2)}$ ratio"
                " & $\\Delta\\rho$ & Loss $\\ell_n$ & $w_n$ & $w_n\\ell_n$ \\\\\n\\midrule\n")
        for j, i in enumerate(idx, 1):
            f.write(f"{j} & {x1[i]:+.2f} & {x2[i]:+.2f} & {k1[i]:.2f} & {k2[i]:.2f} & {dr[i]:+.1f}"
                    f" & {loss[i]:,.2f} & {w[i]:.4f} & {w[i] * loss[i]:,.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


top_table(loss, "top_b.tex")
top_table(w * loss, "top_c.tex")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, d, title in [(axes[0], B, "(b) worst case"), (axes[1], C, "(c) worst weighted")]:
        sel = (k1 == d["k1"]) & (k2 == d["k2"]) & (dr == d["dr"])
        Z = np.full((6, 6), np.nan)
        for i in np.where(sel)[0]:
            Z[X_vals.index(x2[i]), X_vals.index(x1[i])] = loss[i]
        im = ax.imshow(Z, origin="lower", cmap="RdBu_r",
                       vmin=-np.nanmax(abs(Z)), vmax=np.nanmax(abs(Z)))
        ax.set_xticks(range(6), [f"{v:+.2f}" for v in X_vals])
        ax.set_yticks(range(6), [f"{v:+.2f}" for v in X_vals])
        ax.set_xlabel("$X^{(1)}$")
        ax.set_ylabel("$X^{(2)}$")
        ax.set_title(f"{title}: vol ratios {d['k1']:g}, {d['k2']:g}; "
                     f"$\\Delta\\rho$ = {d['dr']:+g}", fontsize=9)
        for a in range(6):
            for b in range(6):
                ax.text(b, a, f"{Z[a, b]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, label="loss")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "heatmap.png"), dpi=180)
except ImportError:
    print("matplotlib not installed; figure skipped")
