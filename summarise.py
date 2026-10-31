import json, math
d = json.load(open("results.json"))
rows = d["summary"]
for name, t in d["truth"].items():
    print(name, "acc=%.3f" % t["acc"], "P(Y=0|111)=%.4f" % (1 - t["post"][7]), "P(Y=1|000)=%.4f" % t["post"][0],
          "P(111)=%.3f" % t["marg"][7], "P(000)=%.3f" % t["marg"][0])
order = ["MV", "PLUG", "CI", "JS1", "JS2"]
for reg in ["independent", "shared_signal", "higher_order"]:
    print("\n==", reg)
    print("m      meth  viol   (se)   FA|acc  FR|rej  pA     pR     pH     loss   acc111 width111")
    for m in sorted({r["m"] for r in rows}):
        for meth in order:
            r = [x for x in rows if x["regime"] == reg and x["m"] == m and x["method"] == meth][0]
            fa = r["fa_rate"]; fr = r["fr_rate"]
            print(f"{m:<6} {meth:<5} {r['viol']:.3f} ({r['viol_se']:.3f}) "
                  f"{'  nan ' if math.isnan(fa) else f'{fa:.3f} '} {'  nan ' if math.isnan(fr) else f'{fr:.3f} '} "
                  f"{r['pA']:.3f}  {r['pR']:.3f}  {r['pH']:.3f}  {r['loss']:.3f}  {r['accept111']:.2f}  "
                  f"{r.get('width111', float('nan')):.3f}")
