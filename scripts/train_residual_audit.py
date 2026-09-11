"""EXP-001b: validation-only tuning followed by sealed-LR report runs.

Reads the trusted, locally generated EXP-001 cache without changing it.
Tuning never evaluates test loss. Each report checkpoint evaluates test once.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from ddm.residual_audit import EVAL_STEPS, FAMILIES, FrozenResidualHead, select_validation_candidate


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def evaluate(head, x, y):
    values = []
    for start in range(0, len(y), 256):
        logits = head(x[start:start+256].cuda())
        values.append(F.cross_entropy(logits, y[start:start+256].cuda(), reduction='none').cpu())
    return torch.cat(values)


def fit(cfg, out, meta, w, x, y, depth, family, seed, lr, phase):
    name = f'{phase}-T{depth}-{family}-seed{seed}-lr{lr:g}'
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    head = FrozenResidualHead(w, cfg['width'], family).cuda()
    initial = {k: v.detach().clone() for k, v in head.named_parameters()}
    native_copy = head.native_weight.clone()
    with torch.no_grad():
        sample = x['train'][:256].cuda()
        native = F.linear(sample, head.native_weight)
        assert torch.equal(head(sample), native), 'step-zero parity failure'
    native_rms = native.square().mean().sqrt().item()
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    history, grads = [], []
    best, selected, best_state = float('inf'), None, None
    for step in range(cfg['max_steps']+1):
        if step:
            idx = torch.randint(len(y['train']), (cfg['batch_size'],), generator=generator)
            loss = F.cross_entropy(head(x['train'][idx].cuda()), y['train'][idx].cuda())
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert head.native_weight.grad is None
            if step in (1, 2, 5, 20):
                ga, gu = head.a.weight.grad.norm().item(), head.u.weight.grad.norm().item()
                grads.append(dict(step=step, grad_a=ga, grad_u=gu, grad_bias=head.a.bias.grad.norm().item()))
                assert (ga == 0) if step == 1 else (ga > 0)
                assert gu > 0
            opt.step()
        if step in cfg['eval_steps']:
            assert torch.equal(head.native_weight, native_copy), 'native head changed'
            val = evaluate(head, x['validation'], y['validation']).double().mean().item()
            assert np.isfinite(val)
            with torch.no_grad():
                norms = {k: v.norm().item() for k, v in head.named_parameters()}
                updates = {k: (v-initial[k]).norm().item() for k, v in head.named_parameters()}
                ratio = head.residual(sample).square().mean().sqrt().item()/native_rms
            history.append(dict(step=step, validation_loss=val, parameter_norms=norms,
                                update_norms=updates, residual_to_native_logit_rms=ratio))
            if val < best:
                best, selected = val, step
                best_state = {k: v.detach().cpu().clone() for k, v in head.named_parameters()}
    assert len(history) == len(EVAL_STEPS)
    assert torch.equal(head.native_weight, native_copy)
    with torch.no_grad():
        for k, v in head.named_parameters():
            v.copy_(best_state[k])
    checkpoint = name+'.pt'
    torch.save(best_state, out/checkpoint)
    row = dict(**meta, phase=phase, loop_depth=depth, head_family=family, seed=seed, lr=lr,
               width=cfg['width'], trainable_parameter_count=sum(p.numel() for p in params),
               head_parameter_count=sum(p.numel() for p in params)+w.numel(),
               initialization='paired A,b by seed; U=0; immutable cached native W',
               optimizer='AdamW', weight_decay=0, lr_schedule='constant', batch_size=cfg['batch_size'],
               steps=cfg['max_steps'], early_stopping=False, dtype='float32', device=torch.cuda.get_device_name(),
               sample_count={s:len(v) for s,v in y.items()}, best_validation_loss=best,
               selected_checkpoint_step=selected, checkpoint_path=checkpoint,
               test_loss=None, history=history, gradients=grads, native_weight_unchanged=True,
               step_zero_max_abs_difference=0., validity_status='passed')
    if phase == 'report':
        values = evaluate(head, x['test'], y['test'])
        assert torch.isfinite(values).all()
        np.save(out/(name+'-test.npy'), values.numpy())
        row.update(test_loss=values.double().mean().item(), test_losses_path=name+'-test.npy')
    row.update(runtime_seconds=time.monotonic()-started,
               peak_memory_bytes=torch.cuda.max_memory_allocated())
    save(out/(name+'.json'), row)
    print(name, 'selected', selected, 'validation', best, flush=True)
    del head, opt, initial, native_copy, best_state
    torch.cuda.empty_cache()
    return row


def summarize(cfg, out, rows, payload, test_idx, meta):
    docs = np.array([payload['doc_ids'][i] for i in test_idx.tolist()])
    unique, inv = np.unique(docs, return_inverse=True)
    counts = np.bincount(inv)
    assert len(unique) == 3125 and np.all(counts == 32)
    by = {(r['loop_depth'],r['seed'],r['head_family']):r for r in rows}
    gains, table, seed_stats = [], [], []
    per_seed = {seed: [] for seed in cfg['report_seeds']}
    native = {d: np.load(out/f'native-T{d}-test.npy') for d in cfg['native_depths']}
    for depth in cfg['native_depths']:
        linear, rich = [], []
        for seed in cfg['report_seeds']:
            a = np.load(out/by[depth,seed,FAMILIES[0]]['test_losses_path']).astype('float64')
            b = np.load(out/by[depth,seed,FAMILIES[1]]['test_losses_path']).astype('float64')
            linear.append(a); rich.append(b)
            per_seed[seed].append(float((a-b).mean()))
        a, b = np.mean(linear, axis=0), np.mean(rich, axis=0)
        gain = a-b; gains.append(gain)
        table.append(dict(depth=depth, native_ce=float(native[depth].mean(dtype='float64')),
                          linear_residual_ce=float(a.mean()), nonlinear_residual_ce=float(b.mean()),
                          G_linear=float((native[depth]-a).mean()), G_rich=float((native[depth]-b).mean()),
                          G_nonlin=float(gain.mean())))
    gains = np.array(gains)
    dm = np.array([np.bincount(inv, weights=g)/counts for g in gains]).T
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boot = np.array([dm[rng.integers(0,len(dm),len(dm))].mean(axis=0)
                     for _ in range(cfg['bootstrap_replicates'])])
    coeff = np.array([-1.5,-.5,.5,1.5])/5
    mean = gains.mean(axis=1)
    for seed, g in per_seed.items():
        seed_stats.append(dict(seed=seed, G_nonlin=g, slope=float(np.dot(g,coeff)), delta=float(g[-1]-g[0])))
    slices = {}
    for name, score in dict(difficulty=native[1], loop_benefit=native[1]-native[4]).items():
        slices[name] = [dict(quartile=i+1,count=len(idx),G_nonlin=gains[:,idx].mean(axis=1).tolist())
                        for i,idx in enumerate(np.array_split(np.argsort(score,kind='stable'),4))]
    save(out/'summary.json', dict(**meta, table=table, per_seed_statistics=seed_stats,
         slope=float(mean@coeff), delta_G_nonlin=float(mean[-1]-mean[0]),
         bootstrap_replicates=cfg['bootstrap_replicates'], bootstrap_seed=cfg['bootstrap_seed'],
         slope_document_bootstrap_95=np.quantile(boot@coeff,[.025,.975]).tolist(),
         delta_document_bootstrap_95=np.quantile(boot[:,-1]-boot[:,0],[.025,.975]).tolist(),
         G_nonlin_document_bootstrap_95=np.quantile(boot,[.025,.975],axis=0).tolist(),
         uncertainty_scope='document resampling conditional on report seeds and selected learning rates',
         secondary_slices=slices, validity_status='completed_post_hoc_primary_width'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--hidden-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg['experiment']=='EXP-001b' and cfg['confirmatory'] is False
    assert cfg['native_depths']==[1,2,3,4] and cfg['families']==list(FAMILIES)
    assert tuple(cfg['eval_steps'])==EVAL_STEPS and cfg['max_steps']==2000
    out, root = Path(args.output), Path(args.hidden_root)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise RuntimeError('Output already initialized; use a new directory to avoid overwriting results')
    save(out/'config.json',cfg)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    (out/'phase.txt').write_text('verify_inputs\n')
    extraction=json.loads((root/'extraction.json').read_text())
    assert sha(root/'hidden.pt') == cfg['hidden_sha256'] == extraction['hidden_sha256']
    payload=torch.load(root/'hidden.pt',map_location='cpu',weights_only=False,mmap=True)
    assert sorted(payload['hidden'])==cfg['native_depths']
    idx={s:torch.tensor([i for i,v in enumerate(payload['splits']) if v==s]) for s in ('train','validation','test')}
    assert [len(idx[s]) for s in idx]==[800000,100000,100000]
    docs={s:{payload['doc_ids'][i] for i in ids.tolist()} for s,ids in idx.items()}
    assert all(not docs[a]&docs[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert all(v.shape==(1000000,2048) and v.dtype==torch.bfloat16 for v in payload['hidden'].values())
    w=payload['native_lm_head_weight'].float()
    assert w.shape==(49152,2048)
    y={s:payload['labels'][ids] for s,ids in idx.items()}
    assert payload['labels'].min()>=0 and payload['labels'].max()<len(w)
    provenance=json.loads((Path(__file__).resolve().parents[1]/'SOURCE_VERSION.json').read_text())
    meta=dict(experiment_id='EXP-001b',run_id=out.name,confirmatory=False,
              parent_run=cfg['parent_run'],hidden_sha256=cfg['hidden_sha256'],
              model=extraction['model'],model_revision=extraction['model_revision'],
              tokenizer_revision=extraction['tokenizer_revision'],dataset=extraction['dataset'],
              dataset_revision=extraction['dataset_revision'],dataset_config=extraction['dataset_config'],
              split='same cached document train/validation/test split as EXP-001 1M',**provenance)
    save(out/'provenance.json',dict(**meta,torch_version=torch.__version__,numpy_version=np.__version__,
                                   cuda_version=torch.version.cuda,device=torch.cuda.get_device_name()))
    tuning=[]; chosen={}
    (out/'phase.txt').write_text('tuning\n')
    for depth in cfg['native_depths']:
        x={s:payload['hidden'][depth][ids].float() for s,ids in idx.items() if s!='test'}
        for family in FAMILIES:
            candidates=[fit(cfg,out,meta,w,x,{s:v for s,v in y.items() if s!='test'},depth,family,
                            cfg['tuning_seed'],lr,'tuning') for lr in cfg['learning_rates']]
            tuning.extend(candidates)
            winner=select_validation_candidate(candidates)
            chosen[f'{depth}/{family}']=dict(lr=winner['lr'],best_validation_loss=winner['best_validation_loss'],
                                            tuning_checkpoint_step=winner['selected_checkpoint_step'])
            save(out/'tuning_metrics.json',tuning)
        del x
    save(out/'selected_learning_rates.json',chosen)
    selected_hash=sha(out/'selected_learning_rates.json')
    (out/'phase.txt').write_text('report\n')
    reports=[]
    for depth in cfg['native_depths']:
        x={s:payload['hidden'][depth][ids].float() for s,ids in idx.items()}
        # A0 is computed only after the entire validation-only LR selection has been sealed.
        native_gpu=w.cuda()
        baseline=lambda z:F.linear(z,native_gpu)
        native=evaluate(baseline,x['test'],y['test'])
        np.save(out/f'native-T{depth}-test.npy',native.numpy())
        del baseline, native_gpu
        for seed in cfg['report_seeds']:
            for family in FAMILIES:
                reports.append(fit(cfg,out,dict(**meta,selected_learning_rates_sha256=selected_hash),
                                   w,x,y,depth,family,seed,chosen[f'{depth}/{family}']['lr'],'report'))
                save(out/'report_metrics.json',reports)
        del x
    assert sha(out/'selected_learning_rates.json')==selected_hash
    summarize(cfg,out,reports,payload,idx['test'],meta)
    (out/'phase.txt').write_text('complete\n')


if __name__=='__main__':
    main()
