"""CPU-only, new fixed diagnostic inputs; not a public benchmark or training corpus."""
import hashlib
import json
from pathlib import Path
import torch
from exllamav3 import Config, Tokenizer
from exllamav3.loader.safetensors_alt import save_file
OUT=Path('/output/capture-quarter-holdout');OUT.mkdir(exist_ok=True)
source=json.loads((OUT/'TEXTS.json').read_text())
config=Config.from_directory('/source');tokenizer=Tokenizer.from_config(config)
raw=Path('/task/c03-input-ids.safetensors').read_bytes()
rows=[];tensors={}
for i,doc in enumerate(source):
    ids=tokenizer.encode(doc['text']).to(torch.int32)
    if doc.get('tokens'):
        assert ids.shape[1]>=doc['tokens'], (doc['id'],ids.shape)
        ids=ids[:,:doc['tokens']].contiguous()
    assert 64<=ids.shape[1]<=8192
    matches=[]
    for j in range(0,ids.shape[1]-63,32):
        window=ids[:,j:j+64].contiguous().numpy().tobytes()
        if window in raw:matches.append(j)
    assert not matches, ('64-token sampled window occurs in c03',doc['id'],matches)
    payload=ids.numpy().tobytes()
    rows.append(dict(id=doc['id'],domain=doc['domain'],tokens=ids.shape[1],token_sha256=hashlib.sha256(payload).hexdigest(),text_sha256=hashlib.sha256(doc['text'].encode()).hexdigest(),overlap_windows_found=matches))
    tensors[f'input_ids.{i}']=ids
save_file(tensors,str(OUT/'input-ids.safetensors'))
(OUT/'DATA.json').write_text(json.dumps(dict(status='prepared',gpu_used=False,rows=rows,total_tokens=sum(r['tokens'] for r in rows),
 input_sha256=hashlib.sha256((OUT/'input-ids.safetensors').read_bytes()).hexdigest(),
 texts_sha256=hashlib.sha256((OUT/'TEXTS.json').read_bytes()).hexdigest(),
 scope='New authored short prompts plus synthetic long Korean incident records and a local C++ source excerpt; sampled64-token windows have no exact byte match anywhere in c03. Not semantic decontamination or general model quality.',
 screen_rule='All finite and matching rotation; flag mean per-document error ratio >1.02 or any document >1.05 per projection. Advisory screen, not automatic full-model acceptance.'),ensure_ascii=False,indent=2))
print([(r['id'],r['tokens']) for r in rows])
