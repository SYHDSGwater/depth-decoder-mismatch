from __future__ import annotations

import argparse, copy, json, math
from pathlib import Path
import torch, yaml
from torch.nn import functional as F
from ddm.heads import LinearRefitHead, ResidualRichHead, parameter_count

ROOT=Path(__file__).resolve().parents[1]

def _load_split(hidden_root:Path,split:str,depth:int):
    path=hidden_root/split/"paired_hidden.pt"
    if not path.exists(): return None
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if depth not in payload["hidden"]: raise KeyError(f"depth {depth} not found in {path}; have {sorted(payload['hidden'])}")
    return payload["hidden"][depth].float(),payload["labels"].long(),payload

@torch.inference_mode()
def _mean_ce(head,x,y,batch_size,device):
    head.eval(); total_nll=0.0; total_n=0
    for start in range(0,len(x),batch_size):
        xb=x[start:start+batch_size].to(device,non_blocking=True); yb=y[start:start+batch_size].to(device,non_blocking=True); loss_sum=F.cross_entropy(head(xb),yb,reduction="sum"); total_nll+=float(loss_sum); total_n+=len(xb)
    head.train(); return total_nll/max(total_n,1)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--depth",type=int,required=True); p.add_argument("--seed",type=int,default=11); p.add_argument("--device",default=None); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); torch.manual_seed(args.seed); hidden_root=ROOT/cfg["hidden_dir"]
    train=_load_split(hidden_root,"train",args.depth); valid=_load_split(hidden_root,"validation",args.depth); test=_load_split(hidden_root,"test",args.depth)
    if train is None or valid is None: raise SystemExit("Confirmatory probe training requires both train and validation hidden splits.")
    x_train,y_train,train_payload=train; x_valid,y_valid,_=valid; native_w=train_payload["native_lm_head_weight"].float(); d=x_train.shape[-1]; v=native_w.shape[0]; kind=cfg["head"]["kind"]
    if kind=="linear_refit": head=LinearRefitHead(d,v,native_w)
    elif kind=="residual_rich": head=ResidualRichHead(d,v,int(cfg["head"]["residual_width"]),native_w)
    else: raise ValueError(f"unknown head kind: {kind}")
    device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"); head=head.to(device); tcfg=cfg["training"]; opt=torch.optim.AdamW(head.parameters(),lr=float(tcfg["lr"]),weight_decay=float(tcfg.get("weight_decay",0.0)))
    bs=int(tcfg["batch_size"]); max_steps=int(tcfg["max_steps"]); eval_every=int(tcfg.get("eval_every",100)); patience_evals=int(tcfg.get("patience_evals",8)); eval_bs=int(tcfg.get("eval_batch_size",bs)); generator=torch.Generator(device="cpu").manual_seed(args.seed)
    best_val=math.inf; best_step=-1; best_state=None; stale=0; history=[]; head.train()
    for step in range(1,max_steps+1):
        idx=torch.randint(0,len(x_train),(min(bs,len(x_train)),),generator=generator); xb=x_train[idx].to(device); yb=y_train[idx].to(device); loss=F.cross_entropy(head(xb),yb); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==1 or step%eval_every==0 or step==max_steps:
            val_ce=_mean_ce(head,x_valid,y_valid,eval_bs,device); train_ce=float(loss.detach()); history.append({"step":step,"train_batch_ce":train_ce,"validation_ce":val_ce})
            if val_ce<best_val-1e-8: best_val=val_ce; best_step=step; best_state=copy.deepcopy(head.state_dict()); stale=0
            else:
                stale+=1
                if stale>=patience_evals: break
    if best_state is None: raise RuntimeError("probe never produced a validation checkpoint")
    head.load_state_dict(best_state); val_ce=_mean_ce(head,x_valid,y_valid,eval_bs,device); test_ce=None
    if test is not None: x_test,y_test,_=test; test_ce=_mean_ce(head,x_test,y_test,eval_bs,device)
    out=ROOT/"outputs"/"probes"/cfg["experiment"]/f"T{args.depth}-{kind}-seed{args.seed}"; out.mkdir(parents=True,exist_ok=True); torch.save(head.state_dict(),out/"head.pt")
    metrics={"depth":args.depth,"kind":kind,"seed":args.seed,"params":parameter_count(head),"best_step":best_step,"validation_ce":val_ce,"test_ce":test_ce,"n_train":len(x_train),"n_validation":len(x_valid),"n_test":0 if test is None else len(test[0]),"selection_rule":"minimum validation cross-entropy; test evaluated only after restoring best validation checkpoint","history":history}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(out)

if __name__=="__main__": main()
