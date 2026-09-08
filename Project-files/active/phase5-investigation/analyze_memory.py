"""Parse archived top samples without rerunning model workloads."""
import datetime
import hashlib
import json
import pathlib
import re
from zoneinfo import ZoneInfo

out=pathlib.Path(__file__).resolve().parent
root=pathlib.Path('/Users/dev-machine/dev/VisionOS/Project-files/active/cache-layout-recovery/evidence')
inputs={
    'journey27':root/'gemma-journey27-20260907/process-top.txt',
    'image-diagnostic':root/'gemma-journey27-20260907/chat-image-diagnostic/process-top.txt',
    'journey28':root/'gemma-journey28-20260907/memory-top.txt',
    'passive-current':out/'passive-memory/top.stdout',
}

def parse(path):
    data=path.read_bytes()
    rows=[]
    for block in data.decode().split('Processes:')[1:]:
        ts=re.search(r'^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)$',block,re.M)
        if not ts: continue
        dt=datetime.datetime.strptime(ts[1],'%Y/%m/%d %H:%M:%S').replace(tzinfo=ZoneInfo('Europe/London'))
        vm=re.search(r'(\d+)\(\d+\) swapins, (\d+)\(\d+\) swapouts',block)
        system=re.search(r'^PhysMem: (.*)$',block,re.M)
        for m in re.finditer(r'^(\d+)\s+(TurboFieldfare\S+)\s+(?:([\d.]+)\s+)?(\d+(?:\.\d+)?)([KMGT])[-+]?\s+',block,re.M):
            rows.append(dict(timestamp_local=ts[1],timestamp_unix_seconds=dt.timestamp(),pid=int(m[1]),process=m[2],
                             cpu_percent=float(m[3]) if m[3] else None,top_mem=m[4]+m[5],mem_mib=float(m[4])*{'K':1/1024,'M':1,'G':1024,'T':1024**2}[m[5]],
                             system_physmem=system[1] if system else None,
                             system_swapins=int(vm[1]) if vm else None,system_swapouts=int(vm[2]) if vm else None))
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),rows=rows)

def summarize(rows):
    r={}
    for pid in sorted({x['pid'] for x in rows}):
        a=[x for x in rows if x['pid']==pid]
        r[pid]=dict(samples=len(a),first=a[0],last=a[-1],minimum_mem_mib=min(x['mem_mib'] for x in a),
                    peak_mem_mib=max(x['mem_mib'] for x in a),peak_samples=[x['timestamp_local'] for x in a if x['mem_mib']==max(v['mem_mib'] for v in a)],
                    system_swapin_pages_delta=a[-1]['system_swapins']-a[0]['system_swapins'],
                    system_swapout_pages_delta=a[-1]['system_swapouts']-a[0]['system_swapouts'])
    return r

all_data={name:parse(path) for name,path in inputs.items()}
(out/'memory-samples.json').write_text(json.dumps(all_data,indent=2)+'\n')
summary={name:summarize(data['rows']) for name,data in all_data.items()}
trace=root/'gemma-journey27-20260907/chat-image-diagnostic/model-trace.jsonl'
events=[json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
steps={}
for r in events:
    steps.setdefault(r['step_id'],{})[r['event']]=r
windows=[]
for step in steps.values():
    if 'input' not in step or 'output' not in step: continue
    start=step['input']['timestamp_unix_seconds']; end=step['output']['timestamp_unix_seconds']
    nearby=[x for x in all_data['image-diagnostic']['rows'] if start-10<=x['timestamp_unix_seconds']<=end+10]
    windows.append(dict(step_index=step['input']['step_index'],start_unix=start,end_unix=end,
                        note='Includes samples up to 10 seconds before and after each model step. Cannot resolve transient vision peak.',samples=nearby))
summary['image-step-windows']=windows
(out/'memory-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,stats in summary.items():
    if name=='image-step-windows':
        print(name, json.dumps(stats,indent=2));continue
    print(name)
    for pid,s in stats.items():
        print(pid,{k:v for k,v in s.items() if k not in ('peak_samples','first','last')},'first',s['first']['timestamp_local'],'last',s['last']['timestamp_local'],'peak first',s['peak_samples'][0])
