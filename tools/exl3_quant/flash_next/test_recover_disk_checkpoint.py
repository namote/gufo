import copy,json
from recover_disk_checkpoint import eligible,functions
r=dict(status='failed',error="AssertionError('Disk reserve below 100 GiB')",cleanup_errors=[],returned_KFD_contexts=[])
e=dict(event='phase_end',text=' -- Forward pass: model.language_model.layers.4',count=2309,current=2308,result='completed')
def log(e):return ' -- GUFO_PROGRESS '+json.dumps(e)+'\n -- Saving checkpoint\n'
eligible(r,log(e),2309)
for change in [dict(error="AssertionError('Host reserve below 8 GiB')"),dict(cleanup_errors=['failed']),dict(returned_KFD_contexts=[123]),dict(status='passed')]:
 try:eligible(dict(r,**change),log(e),2309)
 except AssertionError:pass
 else:raise AssertionError(change)
for change in [dict(result='failed'),dict(current=2307),dict(count=2308)]:
 try:eligible(r,log(dict(e,**change)),2309)
 except AssertionError:pass
 else:raise AssertionError(change)
validate,finish=functions();assert callable(validate) and callable(finish)
print('9 CPU recovery checks passed')
