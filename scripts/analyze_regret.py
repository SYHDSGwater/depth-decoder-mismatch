from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from ddm.metrics import head_regret, ols_slope
ROOT=Path(__file__).resolve().parents[1]

def _from_manual(path):
    data=json.loads(path.read_text(encoding="utf-8")); depths=sorted(int(k) for k in data); regrets=[head_regret(float(data[str(d)]["linear"]),float(data[str(d)]["rich"])) for d in depths]
    return {"depths":depths,"mean_regret":regrets,"ols_slope_of_mean_regret":ols_slope(depths,regrets),"note":"manual aggregate; seed-paired uncertainty unavailable"}

def _from_probe_root(probe_root,split):
    metric_key=f"{split}_ce"; rows=defaultdict(dict)
    for path in sorted(probe_root.glob("T*-*-seed*/metrics.json")):
        m=json.loads(path.read_text(encoding="utf-8")); value=m.get(metric_key)
        if value is None: continue
        kind=m["kind"]; role="linear" if kind=="linear_refit" else "rich" if kind=="residual_rich" else None
        if role is not None: rows[(int(m["depth"]),int(m["seed"]))][role]=float(value)
    paired=[]
    for (depth,seed),vals in sorted(rows.items()):
        if "linear" in vals and "rich" in vals: paired.append({"depth":depth,"seed":seed,"linear_ce":vals["linear"],"rich_ce":vals["rich"],"regret":head_regret(vals["linear"],vals["rich"])})
    if not paired: raise SystemExit(f"No paired linear/rich {split} metrics found under {probe_root}")
    by_depth=defaultdict(list)
    for r in paired: by_depth[r["depth"]].append(r["regret"])
    depths=sorted(by_depth); means=[float(np.mean(by_depth[d])) for d in depths]; stds=[float(np.std(by_depth[d],ddof=1)) if len(by_depth[d])>1 else 0.0 for d in depths]
    by_seed=defaultdict(dict)
    for r in paired: by_seed[r["seed"]][r["depth"]]=r["regret"]
    seed_slopes={str(seed):ols_slope(depths,[vals[d] for d in depths]) for seed,vals in sorted(by_seed.items()) if set(depths).issubset(vals)}
    return {"split":split,"depths":depths,"mean_regret":means,"seed_sd_regret":stds,"ols_slope_of_mean_regret":ols_slope(depths,means),"per_seed_slopes":seed_slopes,"paired_rows":paired,"warning":"probe-seed SD is not the final scientific CI; use document-level bootstrap for confirmatory reporting"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment",required=True); p.add_argument("--probe-root"); p.add_argument("--split",choices=["validation","test"],default="test"); p.add_argument("--loss-json"); p.add_argument("--output"); args=p.parse_args()
    if bool(args.probe_root)==bool(args.loss_json): raise SystemExit("Provide exactly one of --probe-root or --loss-json")
    result=_from_probe_root(ROOT/args.probe_root,args.split) if args.probe_root else _from_manual(Path(args.loss_json)); result["experiment"]=args.experiment; text=json.dumps(result,indent=2); print(text)
    if args.output: path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text+"\n",encoding="utf-8")

if __name__=="__main__": main()
