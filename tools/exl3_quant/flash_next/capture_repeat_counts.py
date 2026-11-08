"""Aggregate integer diagnostics only for an immutable owning gather in one scope."""
import torch

class RepeatedCounts:
    def __init__(self):self.entries={};self.calls=0;self.kernels=0;self.closed=False
    def add(self,x,target,count):
        assert not self.closed
        assert target.dtype==torch.int64 and target.shape==(2,) and target.is_contiguous()
        assert target.device==x.device
        key=(id(x),id(target))
        if key not in self.entries:
            temp=target.new_zeros(target.shape);count(x,temp)
            self.entries[key]=[x,target,temp,0];self.kernels+=1
        item=self.entries[key];assert item[0] is x and item[1] is target
        item[3]+=1;self.calls+=1
    def commit(self):
        assert not self.closed
        for x,target,temp,repetitions in self.entries.values():
            # Integer sums; at most 2M tokens*2560 dimensions*1024 calls < 2^63.
            assert repetitions*x.numel()<2**63
            target.add_(temp*repetitions)
        self.closed=True
