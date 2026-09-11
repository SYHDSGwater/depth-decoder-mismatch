"""Recipe-specific smoke/pilot pipeline; confirmatory defaults remain separate.

Run prepare, extract, probes in order. Model and dataset revisions are pinned in
the config. Outputs must be a fresh run directory. No remote model edits occur.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml
from ddm.adapters.ouro import OuroAdapter
from ddm.heads import LinearRefitHead, ResidualRichHead, parameter_count
from ddm.sampling import document_split, first_window, sample_positions, window_quotas, first_content_occurrence

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def base_meta(cfg):
    version_file = ROOT / 'SOURCE_VERSION.json'
    if version_file.exists():
        provenance = json.loads(version_file.read_text())
    else:
        names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
        digest = hashlib.sha256()
        for name in sorted(set(names)):
            path = ROOT/name
            if path.is_file():
                digest.update(name.encode())
                digest.update(path.read_bytes())
        provenance = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                          source_sha256=digest.hexdigest())
    return dict(experiment_id='EXP-001', stage=cfg['stage'], confirmatory=False,
                git_commit=provenance['git_commit'], source_sha256=provenance['source_sha256'],
                model=cfg['model']['name'], model_revision=cfg['model']['revision'],
                tokenizer_revision=cfg['model']['revision'], dataset=cfg['data']['dataset'],
                dataset_revision=cfg['data']['revision'], dataset_config=cfg['data']['dataset_config'])


def prepare(cfg, model_path, out):
    from transformers import AutoTokenizer
    import fsspec
    import pyarrow.parquet as parquet
    d = cfg['data']
    assert d['positions_per_window'] == 32
    quotas = window_quotas(d['target_count'], d['positions_per_window'])
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/datasets/{d['dataset']}/resolve/{d['revision']}/{d['source_file']}"
    records, seen, counts = [], set(), dict.fromkeys(SPLITS, 0)
    seen_content, duplicate_content_count = set(), 0
    # stream the pinned first sample-10BT shard, do not download the 10BT corpus.
    def source_rows():
        # A synchronous Arrow reader avoids a background scanner finalization
        # crash in the server's datasets/PyArrow combination on early stop.
        with fsspec.open(url, 'rb', block_size=8*1024**2) as handle:
            with parquet.ParquetFile(handle, pre_buffer=False) as reader:
                for batch in reader.iter_batches(batch_size=64, columns=['id','text'], use_threads=False):
                    yield from batch.to_pylist()
    iterator = source_rows()
    for index, row in enumerate(iterator):
        doc_id = str(row.get('id') or hashlib.sha256(row['text'].encode()).hexdigest())
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if d.get('deduplicate_text',False) and not first_content_occurrence(row['text'], seen_content):
            duplicate_content_count += 1
            continue
        split = document_split(doc_id, d['seed'])  # split before tokenization/windowing
        if counts[split] >= quotas[split]:
            continue
        tokens = tok(row['text'], add_special_tokens=False, truncation=False)['input_ids']
        window = first_window(tokens, d['seq_len'])
        if window is None:
            continue
        positions = sample_positions(doc_id, 0, d['seed'], d['seq_len'],
                                     d['positions_per_window'], d['min_position'])
        records.append(dict(document_id=doc_id, source_row=index, window_id=0, split=split,
                            input_ids=window, positions=positions,
                            targets=[window[p+1] for p in positions],
                            text_sha256=hashlib.sha256(row['text'].encode()).hexdigest()))
        counts[split] += 1
        if len(records) % 128 == 0:
            print('PREPARE', counts, flush=True)
        if counts == quotas:
            break
    iterator.close()
    del iterator
    gc.collect()
    assert counts == quotas, counts
    # Content duplicates cannot leak even when source IDs differ.
    content_splits = {}
    for record in records:
        previous = content_splits.setdefault(record['text_sha256'], record['split'])
        assert previous == record['split'], 'duplicate text across document splits'
    path = out / 'windows.jsonl'
    with path.open('x', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    save_json(out / 'sampling.json', dict(**base_meta(cfg), config=d, document_counts=counts,
        sample_counts={s:counts[s]*32 for s in SPLITS}, windows_sha256=file_hash(path),
        duplicate_content_skipped=duplicate_content_count,
        source_url=url, documents_disjoint=True, text_hashes_disjoint=True))


def extract(cfg, model_path, out):
    from transformers import AutoModelForCausalLM
    assert torch.cuda.is_available(), 'CUDA required; never silently fall back to CPU'
    assert not (out / 'hidden.pt').exists(), 'use a fresh output directory'
    # Reject changes to the pinned local config/tokenizer revision before loading.
    for filename in ['config.json', 'configuration_ouro.py', 'modeling_ouro.py', 'tokenizer.json', 'tokenizer_config.json']:
        metadata = Path(model_path)/'.cache/huggingface/download'/f'{filename}.metadata'
        revision, etag, *_ = metadata.read_text().splitlines()
        raw = (Path(model_path)/filename).read_bytes()
        assert revision == cfg['model']['revision'], filename
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == etag, filename
    if cfg['model'].get('weight_sha256'):
        assert file_hash(Path(model_path)/'model.safetensors') == cfg['model']['weight_sha256']
    torch.manual_seed(cfg['data']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=cfg['model']['attention']).cuda().eval()
    model.requires_grad_(False)
    assert model.model.total_ut_steps == model.config.total_ut_steps == 4
    assert model.config.hidden_size == 2048 and model.config.vocab_size == 49152
    adapter = OuroAdapter(model, cfg['model']['native_depths'])
    records = [json.loads(line) for line in (out / 'windows.jsonl').read_text().splitlines()]
    hidden = {depth: [] for depth in (1,2,3,4)}
    labels, splits, doc_ids, positions, window_ids = [], [], [], [], []
    norm_outputs = []
    hook = model.model.norm.register_forward_hook(lambda module, inputs, output: norm_outputs.append(output))
    native_parity = {}
    with torch.inference_mode():
        for index, record in enumerate(records):
            ids = torch.tensor([record['input_ids']], dtype=torch.long, device='cuda')
            pos = torch.tensor(record['positions'], dtype=torch.long, device='cuda')
            norm_outputs.clear()
            result = adapter.extract_depth_hidden(ids, torch.ones_like(ids))
            assert len(norm_outputs) == 4 and set(result.by_depth) == {1,2,3,4}
            for depth, h in result.by_depth.items():
                assert h.shape == (1, 1024, 2048) and torch.isfinite(h).all()
                assert torch.equal(h, norm_outputs[depth-1]), 'wrong final normalization'
                hidden[depth].append(h[0, pos].cpu())
            target = ids[0, pos+1].cpu()
            assert target.tolist() == record['targets'] and int(pos.min()) >= 128 and int(pos.max()) <= 1022
            if index == 0:
                # Wrapper uses zero-based exit_at_step; force each native loop explicitly.
                for depth in (1,2,3,4):
                    expected = model.lm_head(result.by_depth[depth][:, pos]).float()
                    actual = model(ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                   exit_at_step=depth-1, logits_to_keep=pos).logits.float()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    native_parity[str(depth)] = float((actual-expected).abs().max())
                hook.remove()
                # Keep the normalization audit for all later windows as well.
                hook = model.model.norm.register_forward_hook(lambda module, inputs, output: norm_outputs.append(output))
            labels.append(target)
            splits.extend([record['split']]*len(pos))
            doc_ids.extend([record['document_id']]*len(pos))
            positions.extend(record['positions'])
            window_ids.extend([record['window_id']]*len(pos))
            if (index+1) % 128 == 0:
                print('EXTRACT', index+1, '/', len(records), 'seconds', round(time.monotonic()-started,1), flush=True)
        native_w = model.get_output_embeddings().weight.detach().cpu()
    hook.remove()
    payload = dict(hidden={d:torch.cat(xs) for d,xs in hidden.items()}, labels=torch.cat(labels),
                   splits=splits, doc_ids=doc_ids, positions=torch.tensor(positions),
                   window_ids=torch.tensor(window_ids), native_lm_head_weight=native_w)
    assert len(payload['labels']) == cfg['data']['target_count']
    torch.save(payload, out/'hidden.pt')
    save_json(out/'extraction.json', dict(**base_meta(cfg), loop_depth=[1,2,3,4],
        sample_count=cfg['data']['target_count'], split_counts={s:splits.count(s) for s in SPLITS},
        dtype='bfloat16', hidden_storage_dtype='bfloat16', device=torch.cuda.get_device_name(),
        initialization='frozen pretrained backbone', optimizer=None, lr=None, steps=0,
        head_family='native_linear', head_parameter_count=native_w.numel(), seed=cfg['data']['seed'],
        best_validation_loss=None, selected_checkpoint_step=None, test_loss=None,
        runtime_seconds=time.monotonic()-started, peak_memory_bytes=torch.cuda.max_memory_allocated(),
        hidden_sha256=file_hash(out/'hidden.pt'), windows_sha256=file_hash(out/'windows.jsonl'),
        native_logits_max_abs_diff=native_parity, validity_status=f"{cfg['stage']}_extraction_passed"))


@torch.inference_mode()
def losses(head, x, y, batch_size=256):
    head.eval()
    values = []
    for start in range(0,len(y),batch_size):
        values.append(F.cross_entropy(head(x[start:start+batch_size].cuda()),
                                      y[start:start+batch_size].cuda(), reduction='none').cpu())
    return torch.cat(values)


def summarize(cfg, out, rows, native, payload):
    by_pair = {(r['loop_depth'],r['seed'],r['head_family']):r for r in rows}
    test_indices = [i for i,s in enumerate(payload['splits']) if s=='test']
    test_docs = np.array([payload['doc_ids'][i] for i in test_indices])
    docs = sorted(set(test_docs))
    regrets, table = [], []
    for depth in (1,2,3,4):
        seed_arrays = []
        for seed in cfg['training']['seeds']:
            linear = by_pair[depth,seed,'linear_refit']
            rich = by_pair[depth,seed,'residual_rich']
            seed_arrays.append(np.array(linear['test_losses'])-np.array(rich['test_losses']))
        regret = np.mean(seed_arrays,axis=0)
        regrets.append(regret)
        table.append(dict(depth=depth,native_ce=float(np.mean(native[depth])),
            linear_ce=float(np.mean([by_pair[depth,s,'linear_refit']['test_loss'] for s in cfg['training']['seeds']])),
            rich_ce=float(np.mean([by_pair[depth,s,'residual_rich']['test_loss'] for s in cfg['training']['seeds']])),
            regret=float(regret.mean())))
    regrets = np.array(regrets)
    slopes, deltas = [], []
    rng = np.random.default_rng(cfg['analysis']['bootstrap_seed'])
    document_means = np.array([regrets[:,test_docs==doc].mean(axis=1) for doc in docs])
    for _ in range(cfg['analysis']['bootstrap_replicates']):
        mean = document_means[rng.integers(0,len(docs),len(docs))].mean(axis=0)
        slopes.append(float(np.dot(np.array([-1.5,-0.5,0.5,1.5]),mean)/5))
        deltas.append(float(mean[-1]-mean[0]))
    mean_regret = regrets.mean(axis=1)
    slices = {}
    for name,score in {'difficulty':np.array(native[1]),
                       'loop_benefit':np.array(native[1])-np.array(native[4])}.items():
        order = np.argsort(score,kind='stable')
        slices[name] = [dict(quartile=i+1,count=len(idx),mean_regret=regrets[:,idx].mean(axis=1).tolist())
                        for i,idx in enumerate(np.array_split(order,4))]
    seed_statistics = []
    for seed in cfg['training']['seeds']:
        seed_regret = [float(np.mean(np.array(by_pair[d,seed,'linear_refit']['test_losses'])-
                                    np.array(by_pair[d,seed,'residual_rich']['test_losses']))) for d in (1,2,3,4)]
        seed_statistics.append(dict(seed=seed,regret=seed_regret,
            slope=float(np.dot([-1.5,-0.5,0.5,1.5],seed_regret)/5),delta_regret=seed_regret[-1]-seed_regret[0]))
    save_json(out/'summary.json', dict(**base_meta(cfg), validity_status=f"{cfg['stage']}_completed",
        scope='staged exploratory estimate; convergence and confirmatory replication remain separate',
        table=table, slope=float(np.dot([-1.5,-0.5,0.5,1.5],mean_regret)/5),
        delta_regret=float(mean_regret[-1]-mean_regret[0]),
        document_bootstrap_slope_95_percentile=np.quantile(slopes,[0.025,0.975]).tolist(),
        document_bootstrap_delta_95_percentile=np.quantile(deltas,[0.025,0.975]).tolist(),
        per_seed_statistics=seed_statistics,
        uncertainty_scope='document resampling conditional on these three probe seeds; non-confirmatory',
        test_document_count=len(docs), secondary_slices=slices,
        seeds=cfg['training']['seeds'], paired_data_verified=True, extra_capacity_verified=True))


def probes(cfg, out):
    assert torch.cuda.is_available()
    assert not (out/'probe_metrics.json').exists(), 'use a fresh run; do not reevaluate selected tests'
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(out/'hidden.pt',map_location='cpu',weights_only=False)
    extraction = json.loads((out/'extraction.json').read_text())
    assert file_hash(out/'hidden.pt') == extraction['hidden_sha256']
    idx = {s:torch.tensor([i for i,v in enumerate(payload['splits']) if v==s]) for s in SPLITS}
    docs = {s:{payload['doc_ids'][i] for i in idx[s].tolist()} for s in SPLITS}
    assert all(not docs[a]&docs[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    w = payload['native_lm_head_weight'].float()
    labels = {s:payload['labels'][idx[s]] for s in SPLITS}
    t = cfg['training']
    rows, native = [], {}
    for depth in (1,2,3,4):
        x = {s:payload['hidden'][depth][idx[s]].float() for s in SPLITS}
        native_head = LinearRefitHead(w.shape[1],w.shape[0],w).cuda()
        native[depth] = losses(native_head,x['test'],labels['test']).tolist()
        del native_head
        for seed in t['seeds']:
            for kind in ('linear_refit','residual_rich'):
                torch.manual_seed(seed)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                head = (LinearRefitHead(w.shape[1],w.shape[0],w) if kind=='linear_refit'
                        else ResidualRichHead(w.shape[1],w.shape[0],t['residual_width'],w)).cuda()
                with torch.no_grad():
                    sample = x['train'][:8].cuda()
                    torch.testing.assert_close(head(sample), F.linear(sample,w.cuda()),rtol=0,atol=0)
                    if kind=='residual_rich':
                        assert parameter_count(head)>w.numel()
                        assert torch.count_nonzero(head.out_proj.weight)==0
                opt = torch.optim.AdamW(head.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
                generator = torch.Generator().manual_seed(seed)
                best, best_step, best_state, stale, history = float('inf'),0,None,0,[]
                for step in range(1,t['max_steps']+1):
                    head.train()
                    batch_idx = torch.randint(len(x['train']),(t['batch_size'],),generator=generator)
                    loss = F.cross_entropy(head(x['train'][batch_idx].cuda()),labels['train'][batch_idx].cuda())
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    if step==1 or step%t['eval_every']==0 or step==t['max_steps']:
                        val = float(losses(head,x['validation'],labels['validation']).mean())
                        assert np.isfinite(val)
                        history.append(dict(step=step,validation_loss=val,train_batch_loss=float(loss.detach())))
                        if val < best-1e-8:
                            best,best_step,best_state,stale = val,step,copy.deepcopy(head.state_dict()),0
                        else:
                            stale += 1
                        if stale >= t['patience_evals']:
                            break
                assert best_state is not None
                if kind=='residual_rich':
                    assert torch.count_nonzero(head.out_proj.weight)>0, 'residual branch did not learn'
                head.load_state_dict(best_state)
                checkpoint_path = None
                if t.get('save_head_checkpoints',False):
                    checkpoint_path = f'T{depth}-{kind}-seed{seed}.pt'
                    torch.save({k:v.cpu() for k,v in best_state.items()},out/checkpoint_path)
                test = losses(head,x['test'],labels['test'])  # once, after validation selection
                assert torch.isfinite(test).all()
                row = dict(**base_meta(cfg), loop_depth=depth, seed=seed, head_family=kind,
                    head_parameter_count=parameter_count(head), initialization='native W; rich U=0',
                    optimizer='AdamW', lr=t['lr'], steps=step, max_steps=t['max_steps'],
                    lr_schedule='constant', weight_decay=t['weight_decay'], batch_size=t['batch_size'],
                    early_stopping=dict(eval_every=t['eval_every'],patience_evals=t['patience_evals']),
                    dtype='float32', device=torch.cuda.get_device_name(), best_validation_loss=best,
                    selected_checkpoint_step=best_step,test_loss=float(test.mean()), test_losses=test.tolist(),
                    sample_count={s:len(idx[s]) for s in SPLITS}, split='document train/validation/test',
                    hidden_sha256=extraction['hidden_sha256'], runtime_seconds=time.monotonic()-started,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),validity_status=f"{cfg['stage']}_probe_passed",
                    checkpoints_retained=t.get('save_head_checkpoints',False), checkpoint_path=checkpoint_path,
                    history=history)
                rows.append(row)
                save_json(out/'probe_metrics.partial.json',rows)
                print('PROBE',depth,seed,kind,'best_step',best_step,'validation',round(best,4),flush=True)
                del head,opt,best_state
                torch.cuda.empty_cache()
    save_json(out/'probe_metrics.json',rows)
    save_json(out/'native_test_losses.json',native)
    summarize(cfg,out,rows,native,payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','extract','probes'])
    parser.add_argument('--config',default='configs/experiments/exp001-smoke.yaml')
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg['stage'] in ('smoke','pilot','scale_1m') and not cfg['analysis']['confirmatory']
    assert cfg['model']['native_depths'] == [1,2,3,4]
    assert cfg['model']['dtype']=='bfloat16' and cfg['training']['dtype']=='float32'
    assert cfg['training']['optimizer']=='AdamW' and cfg['training']['schedule']=='constant'
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if args.phase == 'prepare':
        w_params = 2048*49152
        rich_params = w_params+cfg['training']['residual_width']*(2048+1+49152)
        estimated = cfg['data']['target_count']*4*2048*2+w_params*2
        if cfg['training'].get('save_head_checkpoints',False):
            estimated += 4*len(cfg['training']['seeds'])*(w_params+rich_params)*4
        assert shutil.disk_usage(out).free > estimated+2*1024**3, 'Insufficient disk space including 2 GiB reserve'
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == cfg, 'configuration changed between phases'
    save_json(out/'config.json',cfg)
    save_json(out/'environment.json',{p:importlib.metadata.version(p) for p in
        ['torch','transformers','datasets','huggingface-hub','tokenizers','numpy','safetensors']})
    if args.phase=='prepare':
        prepare(cfg,args.model_path,out)
    elif args.phase=='extract':
        extract(cfg,args.model_path,out)
    else:
        probes(cfg,out)


if __name__=='__main__':
    main()
