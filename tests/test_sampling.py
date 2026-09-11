from ddm.sampling import document_split, first_window, sample_positions, window_quotas, first_content_occurrence
import pytest


def test_exact_content_duplicates_are_removed_before_id_based_split():
    seen = set()
    records = [('id-1','identical text'),('id-2','different text'),('id-3','identical text')]
    retained = [identifier for identifier,text in records if first_content_occurrence(text,seen)]
    assert retained == ['id-1','id-2']
    assert len(seen)==2


def test_pilot_scales_all_splits_equally():
    smoke = window_quotas(10240)
    pilot = window_quotas(102400)
    assert pilot == {k:10*v for k,v in smoke.items()}
    assert pilot == {'train':2560,'validation':320,'test':320}
    with pytest.raises(ValueError):
        window_quotas(100001)


def test_next_token_targets_and_window_boundary():
    tokens = list(range(2048))
    window = first_window(tokens)
    positions = sample_positions('doc',0,20260911)
    assert len(positions)==len(set(positions))==32
    assert min(positions)>=128 and max(positions)<=1022
    assert all(window[p+1]==p+1 for p in positions)
    assert first_window(tokens[:1023]) is None


def test_sampling_does_not_depend_on_loop_depth_or_iteration_order():
    expected = {doc:sample_positions(doc,0,20260911) for doc in ['a','b','c']}
    assert {doc:sample_positions(doc,0,20260911) for doc in ['c','a','b']}==expected
    assert expected['a'] != expected['b']


def test_document_partition_is_stable_and_disjoint():
    docs = [f'doc-{i}' for i in range(1000)]
    groups = {s:{doc for doc in docs if document_split(doc,20260911)==s}
              for s in ['train','validation','test']}
    assert all(groups.values())
    assert sum(map(len,groups.values()))==1000
    assert not groups['train'] & groups['validation']
    assert not groups['train'] & groups['test']
    assert not groups['validation'] & groups['test']


def test_one_million_targets_have_exact_whole_document_quotas():
    quotas = window_quotas(1000000)
    assert quotas == {'train':25000,'validation':3125,'test':3125}
    assert {k:v*32 for k,v in quotas.items()} == {'train':800000,'validation':100000,'test':100000}
