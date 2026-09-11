"""Frozen EXP-003C control-only tuning and paired recurrent factorial."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch
from torch.nn import functional as F
import yaml
from ddm.tiny_recurrent import TinyRecurrent, decomposition


def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(cfg,out):
    import pyarrow.parquet as pq
    data=out/'data';data.mkdir(exist_ok=True)
    streams={};manifest={}
    for split in ['train','validation']:
        target=data/f'{split}.parquet'
        if not target.exists():
            route=f"datasets/Salesforce/wikitext/resolve/{cfg['dataset_revision']}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
            errors=[]
            for host in ['https://hf-mirror.com/','https://huggingface.co/']:
                try:
                    urllib.request.urlretrieve(host+route,target);break
                except Exception as e:
                    errors.append(str(e))
            else:raise RuntimeError(errors)
        texts=pq.read_table(target,columns=['text'])['text'].to_pylist()
        raw='\n'.join(t for t in texts if t).encode('utf-8')
        assert hashlib.sha256(raw).hexdigest()==cfg[f'{split}_sha256']
        (data/f'{split}.txt').write_bytes(raw)
        streams[split]=torch.tensor(list(raw),dtype=torch.long)
        manifest[split]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),parquet_sha256=sha(target))
    save(out/'data_manifest.json',manifest)
    return streams


def batches(stream,starts):
    seq=stream[starts[:,None]+torch.arange(65)[None,:]]
    return seq[:,:-1].cuda(),seq[:,1:].cuda()


@torch.no_grad()
def evaluate(model,stream,starts):
    model.eval();values=[]
    for indices in starts:
        x,y=batches(stream,indices)
        values.append(torch.stack(decomposition(model(x),y),dim=1).cpu())
    arr=torch.cat(values).double().numpy()
    assert np.isfinite(arr).all()
    assert np.max(np.abs(arr[:,0]-arr[:,1]-arr[:,2]))<1e-5
    return dict(zip(['CE_raw','CE_active','CE_competition','m_dummy'],arr.mean(axis=0))),arr


def fit(cfg,out,streams,meta,depth,size,seed,lr,phase,valstarts):
    name=f'{phase}-T{depth}-V{size}-s{seed}-lr{lr:g}'
    torch.manual_seed(seed);model=TinyRecurrent(depth,size).cuda()
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    # Paired arm initialization is asserted against the 256 model at each fit.
    torch.manual_seed(seed);control=TinyRecurrent(depth,256).cuda()
    for k,v in control.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    with torch.no_grad():
        x,y=batches(streams['validation'],valstarts[0]);model.eval();control.eval()
        assert torch.equal(model.hidden(x),control.hidden(x))
        assert torch.equal(model(x)[...,:256],control(x))
        assert y.min()>=0 and y.max()<256
    del control
    gen=torch.Generator().manual_seed(seed)
    trainstarts=torch.stack([torch.randint(len(streams['train'])-65,(32,),generator=gen) for _ in range(600)])
    batch_hash=hashlib.sha256(trainstarts.numpy().tobytes()).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    def factor(step):
        if step<60:return (step+1)/60
        return .5*(1+math.cos(math.pi*min((step-60)/540,1.)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    history=[];best=float('inf');beststep=None
    for step in range(601):
        if step:
            model.train();x,y=batches(streams['train'],trainstarts[step-1])
            z=model(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten());assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            optimizer.step();scheduler.step()
        if step in cfg['eval_steps']:
            metrics,arr=evaluate(model,streams['validation'],valstarts)
            history.append(dict(step=step,**metrics))
            if metrics['CE_active']<best:best=metrics['CE_active'];beststep=step
    path=name+'.pt';torch.save(model.state_dict(),out/path)
    np.save(out/(name+'-validation.npy'),arr)
    row=dict(**meta,name=name,phase=phase,depth=depth,output_size=size,seed=seed,lr=lr,
             head_family='linear_softmax',parameter_count=sum(p.numel() for p in model.parameters()),
             initialization='public normal(.02) active/body; independent normal(.02) dummy; paired shared tensors',
             optimizer='AdamW',steps=600,supervised_tokens=600*32*64,batch_order_sha256=batch_hash,
             selected_checkpoint_step=600,best_validation_loss=best,best_validation_step=beststep,
             test_loss=None,test_status='not accessed; primary endpoint is fixed validation per card',
             dtype='float32',device=torch.cuda.get_device_name(),history=history,final=history[-1],
             checkpoint=path,validation_arrays=name+'-validation.npy',runtime_seconds=time.monotonic()-start,
             peak_memory_bytes=torch.cuda.max_memory_allocated(),validity_status='paired_checks_passed')
    save(out/(name+'.json'),row);print(name,history[-1],flush=True)
    del model,optimizer;torch.cuda.empty_cache()
    return row


def analyze(rows):
    by={(r['seed'],r['depth'],r['output_size']):r for r in rows};per=[]
    for seed in [1,2,3,4,5]:
        entry=dict(seed=seed)
        for metric in ['CE_active','CE_raw']:
            p=[by[seed,t,4096]['final'][metric]-by[seed,t,256]['final'][metric] for t in [1,4]]
            entry[metric]=dict(P_T1=p[0],P_T4=p[1],interaction=p[1]-p[0])
        per.append(entry)
    summary={}
    for metric in ['CE_active','CE_raw']:
        a=np.array([r[metric]['interaction'] for r in per]);sd=a.std(ddof=1);mean=a.mean();half=2.7764451051977987*sd/math.sqrt(5)
        summary[metric]=dict(mean_interaction=float(mean),seed_sd=float(sd),paired_seed_t95=[float(mean-half),float(mean+half)],
                             dz=float(mean/sd) if sd else None,positive_seeds=int((a>0).sum()))
    return dict(per_seed=per,statistics=summary,uncertainty='two-sided Student t, df=4; paired training seeds; fixed validation sample')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'config.json').exists(),'Use a fresh output directory'
    save(out/'config.json',cfg)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    streams=prepare(cfg,out)
    gen=torch.Generator().manual_seed(cfg['validation_sampling_seed'])
    valstarts=torch.stack([torch.randint(len(streams['validation'])-65,(32,),generator=gen) for _ in range(8)])
    np.save(out/'validation_starts.npy',valstarts.numpy())
    meta=dict(experiment='EXP-003C',model='Murugan compact shared-body Transformer D32 L4 H4 FF128',
              dataset='WikiText-2 raw UTF-8 bytes',dataset_revision=cfg['dataset_revision'],tokenizer='identity byte IDs 0..255',
              train_sha256=cfg['train_sha256'],validation_sha256=cfg['validation_sha256'],
              validation_starts_sha256=hashlib.sha256(valstarts.numpy().tobytes()).hexdigest(),
              **json.loads((Path(__file__).resolve().parents[1]/'SOURCE_VERSION.json').read_text()))
    save(out/'environment.json',dict(torch=torch.__version__,device=torch.cuda.get_device_name(),cuda=torch.version.cuda))
    (out/'phase.txt').write_text('tuning')
    tuning=[];selected={}
    for t in [1,4]:
        candidates=[fit(cfg,out,streams,meta,t,256,83,lr,'tuning',valstarts) for lr in cfg['lr_grid']]
        tuning+=candidates;selected[str(t)]=min(candidates,key=lambda r:r['final']['CE_active'])['lr']
        save(out/'tuning.json',tuning)
    save(out/'selected_lrs.json',selected)
    (out/'phase.txt').write_text('primary')
    rows=[]
    for seed in [1,2,3,4,5]:
        for t in [1,4]:
            for v in [256,4096]:
                rows.append(fit(cfg,out,streams,meta,t,v,seed,selected[str(t)],'primary',valstarts));save(out/'primary.json',rows)
    assert len({r['batch_order_sha256'] for r in rows})==5
    result=analyze(rows);save(out/'summary.json',dict(**meta,**result,selected_lrs=selected))
    active=result['statistics']['CE_active']
    # Card requires a secondary per-arm LR check only for a positive primary mean.
    if active['mean_interaction']>0:
        (out/'phase.txt').write_text('secondary_per_arm_tuning')
        secondary=[];robust_lrs={}
        for t in [1,4]:
            candidates=[fit(cfg,out,streams,meta,t,4096,83,lr,'secondary_tuning',valstarts) for lr in cfg['lr_grid']]
            secondary+=candidates;robust_lrs[str(t)]=min(candidates,key=lambda r:r['final']['CE_active'])['lr']
        save(out/'secondary_tuning.json',secondary);save(out/'secondary_lrs.json',robust_lrs)
        robust=[r for r in rows if r['output_size']==256]
        for seed in [1,2,3,4,5]:
            for t in [1,4]:
                if robust_lrs[str(t)]==selected[str(t)]:robust.append(next(r for r in rows if r['seed']==seed and r['depth']==t and r['output_size']==4096))
                else:robust.append(fit(cfg,out,streams,meta,t,4096,seed,robust_lrs[str(t)],'secondary_report',valstarts))
        save(out/'secondary_report.json',robust);save(out/'secondary_summary.json',analyze(robust))
    (out/'phase.txt').write_text('complete')


if __name__=='__main__':main()
