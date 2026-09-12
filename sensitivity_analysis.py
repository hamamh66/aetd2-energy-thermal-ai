"""
Sensitivity analysis of the AET-D2 decision layer (revision R2, Reviewer 3, Comments 1-2).

Re-evaluates the policy-guided action assignment on the 1000 evaluated samples
(aetd2_results_first_1000.csv) while perturbing (i) the utility-score coefficients
and (ii) the thermal-proxy parameters. Ensemble energy predictions are held fixed.

Outputs: sensitivity_utility_weights.csv, sensitivity_thermal_proxy.csv,
         sensitivity_summary.csv, sensitivity_action_share.png
"""
import numpy as np, pandas as pd, itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "aetd2_results_first_1000.csv")
EPS = 1e-6

# ---- nominal parameters (exactly as in AETD2_notebook.ipynb) ----------------
NOM_THERMAL = dict(T_SET_HEAT=21.0, T_SET_COOL=24.0, OCC_DAY=1.0, OCC_NIGHT=0.6,
                   K_HEAT=120.0, K_COOL=140.0, K_INERTIA=40.0, LAMBDA=0.6)
NOM_UTIL = {
    "Energy Storage":   dict(w_esi=1.5, w_surplus=0.0004, w_gap=-0.2),
    "Normal Operation": dict(w_esi=1.2, w_imb=-0.0002, w_gap=-0.1),
    "HVAC Moderation":  dict(w_esi=0.8, w_gap=0.4, w_E=-0.0001),
    "Load Reduction":   dict(w_esi=0.9, w_T=0.0002, w_gap=0.2),
}

def thermal_proxy(p):
    occ = np.where((d.hour >= 7) & (d.hour <= 22), p["OCC_DAY"], p["OCC_NIGHT"])
    heat = np.maximum(0, p["T_SET_HEAT"] - d.temp) * occ
    cool = np.maximum(0, d.temp - p["T_SET_COOL"]) * occ
    inertia = p["LAMBDA"] * (pd.Series(heat).shift(1).fillna(0) + pd.Series(cool).shift(1).fillna(0))
    # the lag of the first evaluated sample lies outside the 1000-sample window: reuse the stored value
    inertia.iloc[0] = d.thermal_inertia_proxy.iloc[0] * (p["LAMBDA"] / NOM_THERMAL["LAMBDA"])
    gap = np.where(d.temp < p["T_SET_HEAT"], p["T_SET_HEAT"] - d.temp,
                   np.where(d.temp > p["T_SET_COOL"], d.temp - p["T_SET_COOL"], 0))
    T = p["K_HEAT"] * heat + p["K_COOL"] * cool + p["K_INERTIA"] * inertia.values
    return T, gap

def decide(E, T, gap, u):
    esi = E / (E + T + EPS)
    s = {
        "Energy Storage":   u["Energy Storage"]["w_esi"] * esi + u["Energy Storage"]["w_surplus"] * np.maximum(E - T, 0) + u["Energy Storage"]["w_gap"] * gap,
        "Normal Operation": u["Normal Operation"]["w_esi"] * esi + u["Normal Operation"]["w_imb"] * np.abs(E - T) + u["Normal Operation"]["w_gap"] * gap,
        "HVAC Moderation":  u["HVAC Moderation"]["w_esi"] * (1 - esi) + u["HVAC Moderation"]["w_gap"] * gap + u["HVAC Moderation"]["w_E"] * E,
        "Load Reduction":   u["Load Reduction"]["w_esi"] * (1 - esi) + u["Load Reduction"]["w_T"] * T + u["Load Reduction"]["w_gap"] * gap,
    }
    S = pd.DataFrame(s)
    return S.idxmax(axis=1)

def shares(actions):
    v = actions.value_counts(normalize=True) * 100
    return {a: round(v.get(a, 0.0), 1) for a in NOM_UTIL}

E = d.Predicted_Energy_Wh.values

# ---- 0. reproduce nominal -----------------------------------------------------
T0, g0 = thermal_proxy(NOM_THERMAL)
assert np.allclose(T0, d.Thermal_Load_Wh.values, atol=1e-6), "thermal proxy not reproduced"
assert np.allclose(g0, d.Comfort_Gap.values), "comfort gap not reproduced"
a0 = decide(E, T0, g0, NOM_UTIL)
assert (a0.values == d.Selected_Action.values).all(), "decisions not reproduced"
nominal = shares(a0)
print("Nominal:", nominal)

rows = []
def record(family, label, actions, **extra):
    r = dict(family=family, perturbation=label, **shares(actions), **extra)
    rows.append(r); return r

# ---- 1. utility-weight perturbations (one-at-a-time, +/-25%, +/-50%) -----------
for act, coefs in NOM_UTIL.items():
    for k in coefs:
        for f in (0.5, 0.75, 1.25, 1.5):
            u = {a: dict(c) for a, c in NOM_UTIL.items()}
            u[act][k] = coefs[k] * f
            record("utility", f"{act}:{k} x{f}", decide(E, T0, g0, u))

# joint scaling of all comfort-gap coefficients and all ESI coefficients
for f in (0.5, 0.75, 1.25, 1.5, 2.0):
    u = {a: {k: (v * f if k == "w_gap" else v) for k, v in c.items()} for a, c in NOM_UTIL.items()}
    record("utility", f"all w_gap x{f}", decide(E, T0, g0, u))
    u = {a: {k: (v * f if k == "w_esi" else v) for k, v in c.items()} for a, c in NOM_UTIL.items()}
    record("utility", f"all w_esi x{f}", decide(E, T0, g0, u))

# ---- 2. thermal-proxy perturbations -------------------------------------------
for k in ("K_HEAT", "K_COOL", "K_INERTIA", "LAMBDA"):
    for f in (0.5, 0.75, 1.25, 1.5):
        p = dict(NOM_THERMAL); p[k] = NOM_THERMAL[k] * f
        T, g = thermal_proxy(p)
        record("thermal", f"{k} x{f}", decide(E, T, g, NOM_UTIL), mean_T=round(T.mean(), 1))
# joint scaling of all thermal weights (equivalent to rescaling C_th)
for f in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0):
    p = dict(NOM_THERMAL)
    for k in ("K_HEAT", "K_COOL", "K_INERTIA"): p[k] = NOM_THERMAL[k] * f
    T, g = thermal_proxy(p)
    record("thermal", f"all K x{f}", decide(E, T, g, NOM_UTIL), mean_T=round(T.mean(), 1))
# set-points and occupancy
for th, tc in [(19, 26), (20, 25), (20, 24), (21, 25), (22, 23), (22, 26), (18, 27)]:
    p = dict(NOM_THERMAL, T_SET_HEAT=float(th), T_SET_COOL=float(tc))
    T, g = thermal_proxy(p)
    record("thermal", f"setpoints ({th},{tc}) degC", decide(E, T, g, NOM_UTIL), mean_T=round(T.mean(), 1))
for on in (0.3, 0.8, 1.0):
    p = dict(NOM_THERMAL, OCC_NIGHT=on)
    T, g = thermal_proxy(p)
    record("thermal", f"OCC_NIGHT={on}", decide(E, T, g, NOM_UTIL), mean_T=round(T.mean(), 1))

# ---- 3. treatment of negative ensemble predictions ----------------------------
record("robustness", "clip E_hat at 0", decide(np.maximum(E, 0), T0, g0, NOM_UTIL))

# ---- 4. simple threshold rule on ETSI = T/E (for comparison) -------------------
etsi = T0 / (np.maximum(E, 0) + EPS)
for t1, t2 in [(0.5, 1.0), (0.5, 1.5), (0.8, 1.2), (0.25, 1.0), (0.5, 2.0)]:
    a = pd.Series(np.where(etsi < t1, "Energy Storage", np.where(etsi <= t2, "Normal Operation", "HVAC Moderation")))
    record("etsi-threshold", f"tau1={t1}, tau2={t2}", a)

res = pd.DataFrame(rows)
res.to_csv(HERE / "sensitivity_all.csv", index=False)
res[res.family == "utility"].to_csv(HERE / "sensitivity_utility_weights.csv", index=False)
res[res.family == "thermal"].to_csv(HERE / "sensitivity_thermal_proxy.csv", index=False)

# summary: range of the load-adaptation share within each family
summ = res.groupby("family")["HVAC Moderation"].agg(["min", "max", "count"]).reset_index()
summ.to_csv(HERE / "sensitivity_summary.csv", index=False)
print(summ)
print(res.to_string())
