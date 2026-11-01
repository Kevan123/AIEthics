"""Print the values reported in the manuscript tables from results.json."""
import json, math

d = json.load(open("results.json"))
rows = d["summary"]
REGIMES = ["independent", "shared_signal", "higher_order"]
MS = sorted({r["m"] for r in rows})


def get(reg, m, meth, key):
    for r in rows:
        if r["regime"] == reg and r["m"] == m and r["method"] == meth:
            return r[key]
    return float("nan")


def f(x, nd=3):
    return "  nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


print("== True probabilities and oracle")
for reg in REGIMES:
    t = d["truth"][reg]
    print(f"{reg:14s} P(Y=0|111)={1 - t['post'][7]:.4f} P(Y=1|000)={t['post'][0]:.4f}")
    for c in ("base", "lowref"):
        o = t["oracle_" + c]
        print(f"   oracle {c:6s} referral={o['pH']:.3f} loss={o['loss']:.3f} accept111={o['accept111']} reject000={o['reject000']}")

print("\n== Table 1: breach rate (base costs)")
meths = ["MV", "PLUG", "CI", "JS1", "JS2", "DC"]
print("regime         m      " + "  ".join(f"{x:>5s}" for x in meths))
for reg in REGIMES:
    for m in MS:
        print(f"{reg:14s} {m:<6d} " + "  ".join(f(get(reg, m, x, "viol")) for x in meths))

print("\n== Table 2: referral / accept111 / reject000 / loss (base costs)")
for reg in REGIMES:
    print(f"-- {reg}")
    for m in MS:
        line = f"{m:<6d}"
        for x in ["JS1", "JS2", "DC", "CI"]:
            line += f" | {x}: ref {f(get(reg, m, x, 'pH'))} a111 {f(get(reg, m, x, 'accept111'), 2)} r000 {f(get(reg, m, x, 'reject000'), 2)} loss {f(get(reg, m, x, 'loss'))}"
        print(line)

print("\n== Table 3: low referral cost (c_H = 0.2)")
for reg in REGIMES:
    print(f"-- {reg}")
    for m in MS:
        line = f"{m:<6d}"
        for x in ["JS1_lowref", "JS2_lowref", "DC_lowref"]:
            line += f" | {x}: viol {f(get(reg, m, x, 'viol'))} ref {f(get(reg, m, x, 'pH'))} a111 {f(get(reg, m, x, 'accept111'), 2)} loss {f(get(reg, m, x, 'loss'))}"
        print(line)

print("\n== Mean seconds per replication")
for x in ["JS1", "JS2", "DC"]:
    v = [r["secs"] for r in rows if r["method"] == x]
    print(f"{x}: {sum(v) / len(v):.4f}")

print("\n== Max breach over all cells, certification methods")
for x in ["JS1", "JS2", "DC", "JS1_lowref", "JS2_lowref", "DC_lowref"]:
    print(x, max(r["viol"] for r in rows if r["method"] == x))
