"""Keep a conditional indexer from owning the always-shared QKV input Hessian."""
import functools

def install(attention_class):
    original=attention_class.decode_flash_attn_nc
    @functools.wraps(original)
    def fixed(self,x,bsz,seqlen,params):
        if 'capture' in params and self.qsa_indexer is not None:
            owner=self.q_proj;conditional=self.qsa_indexer.index_qk_proj
            assert owner.qmap and owner.qmap==conditional.qmap
            assert owner.in_features==conditional.in_features and owner.device==conditional.device
            shared=params['capture']
            if owner.qmap not in shared:
                shared[owner.qmap]=owner.init_H_data(True)
            assert shared[owner.qmap]['first_key']==owner.key,'Mixed capture owner history'
        return original(self,x,bsz,seqlen,params)
    attention_class.decode_flash_attn_nc=fixed
    def restore():
        assert attention_class.decode_flash_attn_nc is fixed
        attention_class.decode_flash_attn_nc=original
    return restore
