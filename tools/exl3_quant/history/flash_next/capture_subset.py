"""Uninstalled candidate: keep every checkpoint row; run capture on fixed indices.

Only the pinned converter's discarded all-expert forward may be skipped. First
five normal reference forwards, quant_preserves plumbing and full state advance
remain the unchanged converter's responsibility. No input/checkpoint reindexing.
"""
import hashlib
import inspect
import sys
import capture_fast


def selection(index, selected, expected_rows, state, original_ids, params):
    assert type(index) is int and 0<=index<expected_rows
    assert len(state)==len(original_ids)==expected_rows
    assert params['input_ids'] is original_ids[index], 'Original row/ID mapping changed'
    return index in selected


def install(cm, selected_indices, expected_rows, expected_tokens):
    from exllamav3.modules.transformer import TransformerBlock
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    body=inspect.unwrap(cm.main)
    assert hashlib.sha256(inspect.getsource(body).encode()).hexdigest()==capture_fast._MAIN_SHA
    assert list(selected_indices)==sorted(set(selected_indices)) and selected_indices
    assert all(type(i) is int and 0<=i<expected_rows for i in selected_indices)
    assert expected_tokens>0
    selected=frozenset(selected_indices);stats={}
    # One wrapper owns automatic scope detection; keep the original skip-down
    # implementation in explicit mode so the extra frame cannot disable it.
    restore_fast=capture_fast.install(cm,automatic=False)
    original=TransformerBlock.forward
    def forward(block,x,params,out_dtype=None):
        frame=sys._getframe(1)
        discarded=(frame.f_code is body.__code__ and 'capture' in params
                   and params.get('activate_all_experts') is True and type(block.mlp) is BlockSparseMLP)
        if not discarded:
            return original(block,x,params,out_dtype)
        local=frame.f_locals
        assert local['bad_rows']==set()
        keep=selection(local['i'],selected,expected_rows,local['state'],local['original_input_ids'],params)
        # Validate even skipped rows: a future non-terminal/sliced block must not
        # silently bypass the structural safety checks for discarded outputs.
        capture_fast._targets(block,params)
        entry=stats.setdefault(block.key,dict(rows=0,selected_rows=0,selected_tokens=0,skipped_rows=0))
        assert local['i']==entry['rows'], 'Capture row order changed'
        entry['rows']+=1
        if keep:
            entry['selected_rows']+=1;entry['selected_tokens']+=params['input_ids'].shape[-1]
            with capture_fast._discarded_capture(block,params):
                return original(block,x,params,out_dtype)
        entry['skipped_rows']+=1
        # The exact pinned call site discards this result and recomputes each
        # normal reference with capture/all-expert disabled. Preserve the input.
        return x
    TransformerBlock.forward=forward
    restored=False
    def restore():
        nonlocal restored
        if restored:return
        assert TransformerBlock.forward is forward
        TransformerBlock.forward=original;restore_fast();restored=True
    def validate_complete():
        assert stats, 'No supported capture block was observed'
        for entry in stats.values():
            assert entry['rows']==expected_rows
            assert entry['selected_rows']==len(selected)
            assert entry['selected_tokens']==expected_tokens
            assert entry['skipped_rows']==expected_rows-len(selected)
    return restore,validate_complete,stats
