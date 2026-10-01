"""Decision map in the settled state at the Section 5 constants (Figure 1)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "pdf.fonttype": 42})
m, h, C, cw, q, tau = 0.35, 1.0, 0.5, 0.006, 0.75, 10
rho = 1 - (1 - q) ** (1 / tau)
mt = m * h / (m + h); pih = m / (m + h)
floor = C / (q * m); vstar = C / (q * (mt - cw / rho))
v = np.geomspace(vstar, 2000, 400)
pgv = (m * (1 - q) + C / v + cw * q / rho) / ((m + h) - q * m)
prv = 1 - (C / v + cw * q / rho) / (q * m)
lim_rv = 1 - cw / (rho * m); lim_gv = (m * (1 - q) + cw * q / rho) / ((m + h) - q * m)
x0, x1 = 0.5, 2000
fig, ax = plt.subplots(figsize=(5.09, 3.23))
ax.set_xscale("log"); ax.set_xlim(x0, x1); ax.set_ylim(0, 1)
ax.fill_between([x0, x1], pih, 1, color="0.93", lw=0)
ax.fill_between([x0, x1], 0, pih, color="0.84", lw=0)
ax.fill_between(v, pgv, prv, color="0.88", lw=0)
ax.plot(v, prv, color="0.3", lw=1.6); ax.plot(v, pgv, color="0.3", lw=1.6)
ax.plot([x0, vstar], [pih, pih], color="0.3", lw=1.0)
for y in (lim_rv, lim_gv, pih):
    ax.axhline(y, color="0.45", lw=0.6, ls=":")
for xv in (floor, vstar):
    ax.axvline(xv, color="0.15", lw=0.8, ls="--")
ax.text(x0 * 1.06, 0.965, f"floor = \\${floor:.2f}", ha="left", va="top")
ax.text(vstar * 1.07, 0.965, f"$v^*$ = \\${vstar:.2f}", ha="left", va="top")
ax.text(0.8, 0.62, "reject", ha="center", color="0.45", fontsize=11)
ax.text(0.75, 0.11, "grant", ha="center", fontsize=11)
ax.text(120, 0.52, "verify", ha="center", fontsize=11)
ax.text(x0 * 1.08, pih + 0.015, r"$\hat\pi_F = m/(m+h)$", ha="left", va="bottom", fontsize=8)
ax.text(x1 / 1.1, lim_rv + 0.012, r"$\pi_{rv} \to 1 - c_w/(\rho m)$", ha="right", va="bottom", fontsize=8)
ax.text(x1 / 1.1, lim_gv - 0.012, r"$\pi_{gv} \to (m(1-q)+c_w q/\rho)/((m+h)-qm)$", ha="right", va="top", fontsize=8)
# worked example of Section 3: $30 and $1.50 at score 0.15
ax.plot([30], [0.15], marker="o", ms=5, mfc="white", mec="0.1", mew=1.0)
ax.plot([1.5], [0.15], marker="s", ms=4.5, mfc="white", mec="0.1", mew=1.0)
ax.annotate("\\$30: verify", (30, 0.15), xytext=(30, 0.32), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
ax.annotate("\\$1.50: grant", (1.5, 0.15), xytext=(1.6, 0.185), ha="right", fontsize=8)
ax.set_xlabel(r"exposure $v$ (dollars, log scale)"); ax.set_ylabel(r"suspicion $\pi$")
fig.tight_layout(); fig.savefig("figs/fig7_decision_map.pdf", metadata={"CreationDate": None})
print(f"rho={rho:.4f} floor={floor:.3f} vstar={vstar:.3f} pih={pih:.4f} lim_rv={lim_rv:.4f} lim_gv={lim_gv:.4f}")
