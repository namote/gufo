import collections
import hashlib
import json
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.modules import Linear

config=Config.from_directory('/source')
models={'main':Model.from_config(config)}
for component in ['mtp','vision']:
    if component in config.model_classes:
        models[component]=Model.from_config(config,component=component)
linears=[];modules=[]
for component,model in models.items():
    for i,top in enumerate(model.modules):
        modules.append(dict(component=component,index=i,key=top.key,type=type(top).__name__,slices=top.num_slices))
        for m in top:
            if isinstance(m,Linear):
                linears.append(dict(component=component,key=m.key,qmap=m.qmap,qbits_key=m.qbits_key,
                       in_features=m.in_features,out_features=m.out_features,numel=m.weights_numel()))
Path('/output/model-inventory.json').write_text(json.dumps({'modules':modules,'linears':linears},indent=2))
print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'linears':len(linears),
                  'modules':modules[:7],'roles':dict(collections.Counter((l['qbits_key'] or 'none') for l in linears))}))
