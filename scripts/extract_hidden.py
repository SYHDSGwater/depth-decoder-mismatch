from __future__ import annotations

"""Model-facing extraction entrypoint.

This script intentionally requires an explicit dataset choice rather than hiding
one in the repo. It extracts paired token examples at every configured depth.
The first milestone is a smoke test; large-scale storage/sharding can be added
only after EXP-001 validity is verified.
"""

import argparse, random
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parents[1]

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--split",default="validation"); p.add_argument("--text-file",help="UTF-8 file with one document per line; explicit local smoke-test data"); args=p.parse_args()
    cfg=load_cfg(args.config)
    if not args.text_file: raise SystemExit("Pass --text-file for the initial reproducible smoke test; dataset integration is intentionally not implicit.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ddm.adapters.ouro import OuroAdapter
    from ddm.adapters.nanbeige import NanbeigeAdapter
    mcfg=cfg["model"]; dtype=getattr(torch,mcfg.get("dtype","bfloat16"))
    tok=AutoTokenizer.from_pretrained(mcfg["name"],revision=mcfg.get("revision"),trust_remote_code=mcfg.get("trust_remote_code",True))
    model=AutoModelForCausalLM.from_pretrained(mcfg["name"],revision=mcfg.get("revision"),trust_remote_code=mcfg.get("trust_remote_code",True),torch_dtype=dtype,device_map="auto").eval()
    name=mcfg["name"].lower(); adapter=OuroAdapter(model,mcfg["native_depths"]) if "ouro" in name else NanbeigeAdapter(model,mcfg["native_depths"])
    docs=[x.rstrip("\n") for x in Path(args.text_file).read_text(encoding="utf-8").splitlines() if x.strip()]; rnd=random.Random(cfg["data"].get("seed",17)); docs=docs[:cfg["data"].get("max_sequences",len(docs))]
    records={int(d):[] for d in mcfg["native_depths"]}; labels_all=[]; doc_ids_all=[]; pos_all=[]; native_lm_head_weight=None
    for doc_id,text in enumerate(docs):
        batch=tok(text,return_tensors="pt",truncation=True,max_length=cfg["data"].get("max_length",512)); input_ids=batch["input_ids"].to(model.device); attn=batch.get("attention_mask"); attn=attn.to(model.device) if attn is not None else None
        if input_ids.shape[1]<2: continue
        out=adapter.extract_depth_hidden(input_ids,attn)
        if native_lm_head_weight is None: native_lm_head_weight=out.native_lm_head_weight.to(dtype=torch.float16,device="cpu")
        candidates=list(range(input_ids.shape[1]-1)); k=min(len(candidates),cfg["data"].get("sample_tokens_per_sequence",64)); positions=sorted(rnd.sample(candidates,k)); labels=input_ids[0,torch.tensor(positions,device=input_ids.device)+1].cpu()
        for d,h in out.by_depth.items(): records[int(d)].append(h[0,positions].to(dtype=torch.float16,device="cpu"))
        labels_all.append(labels); doc_ids_all.extend([doc_id]*k); pos_all.extend(positions)
    if not labels_all or native_lm_head_weight is None: raise SystemExit("No usable token pairs were extracted; check the input text and max_length.")
    outdir=ROOT/cfg["storage"]["output_dir"]/args.split; outdir.mkdir(parents=True,exist_ok=True)
    payload={"hidden":{d:torch.cat(xs,0) for d,xs in records.items()},"labels":torch.cat(labels_all,0),"doc_ids":torch.tensor(doc_ids_all,dtype=torch.int32),"positions":torch.tensor(pos_all,dtype=torch.int32),"meta":{"model":mcfg["name"],"depths":list(records),"paired":True},"native_lm_head_weight":native_lm_head_weight}
    torch.save(payload,outdir/"paired_hidden.pt"); print(outdir/"paired_hidden.pt")

if __name__=="__main__": main()
