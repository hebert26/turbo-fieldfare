#!/usr/bin/env python3
"""Main-only bounded probe. Pinned build and guards, own-child memory watchdog."""
import argparse,datetime,json,os,pathlib,re,subprocess,time,uuid
from evidence_helpers import BASE,sha,overlay_hashes
ROOT=next(p for p in BASE.parents if (p/'.git').exists())
FLAGS={'TURBO_QWEN_SOURCE_VALIDATION_FAST':'1','TURBO_QWEN_EXPERT_CACHE_RESIDENCY':'1','TURBO_QWEN_GPU_LINEAR_PREPARATION':'1','TURBO_QWEN_GROUPED_LINEAR_PREFILL':'1','TURBO_QWEN_EXPERT_PROJECTION_BATCH':'0','TURBO_QWEN_SOURCE_MEMBERSHIP_SCAN':'0','TURBO_QWEN_EXPERT_READ_KNOWN_NONE_4':'0','TURBO_QWEN_EXACT_TOKEN_CAPTURE':'0'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration');args=parser.parse_args()
    run=BASE/('run-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);run.mkdir()
    report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'commands':[],'environment':FLAGS,'memorySamplingSeconds':2,'wholeRunTimeoutSeconds':420}
    child=None
    def command(argv,label):
        value=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
        (run/(label+'.stdout')).write_text(value.stdout);(run/(label+'.stderr')).write_text(value.stderr)
        report['commands'].append({'argv':argv,'exit':value.returncode});return value
    def stop_owned(reason):
        report['watchdogAbort']=reason
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait();report['forcedKillOwnChild']=True
    try:
        frozen=json.loads((BASE/'frozen-hashes.json').read_text())
        for name,digest in frozen.items():assert sha(BASE/name)==digest,'stage drift: '+name
        receipt=json.loads((BASE/'build-receipt.json').read_text());assert receipt['exit']==0 and receipt['overlayStable'],'successful stable build required'
        assert receipt['stageHashesSHA256']==sha(BASE/'frozen-hashes.json'),'build/stage mismatch'
        assert receipt['overlayHashesSHA256']==sha(BASE/'overlay-after-build-hashes.json'),'build/source receipt drift'
        assert overlay_hashes()==json.loads((BASE/'overlay-after-build-hashes.json').read_text()),'overlay/dependency drift'
        binary=BASE/'build/release/TurboFieldfareExpertPredictRecall';assert binary.is_file() and sha(binary)==receipt['binarySHA256'],'built binary drift'
        pre=command(['python3',str(BASE/'preflight.py')],'preflight');assert pre.returncode==0,'preflight failed'
        busy=command(['pgrep','-fl','TurboFieldfareExactMetadataCost|TurboFieldfareExactMetadataParallel|TurboFieldfareExactBlockProbe|TurboFieldfareExpertPredictRecall|resident-kernel-timing|source64-token4|qwen.*harness'],'extra-process-guard');assert busy.returncode==1 and not busy.stdout.strip(),'competing harness'
        hardware=command(['system_profiler','SPHardwareDataType'],'hardware');assert hardware.returncode==0
        env={k:v for k,v in os.environ.items() if not k.startswith(('TURBO_QWEN_','TURBOFIELDFARE_'))};env.update(FLAGS)
        commandLine=[str(binary),str(pathlib.Path(args.registration).resolve()),str(BASE/'request.json'),str(run/'result.json')]
        report.update(binarySHA256=sha(binary),stageHashes=frozen,probeCommand=commandLine)
        with (run/'probe.stdout').open('wb') as out,(run/'probe.stderr').open('wb') as err,(run/'memory.jsonl').open('w') as memory:
            child=subprocess.Popen(commandLine,cwd=ROOT,stdout=out,stderr=err,env=env)
            report['ownedPID']=child.pid;start=time.monotonic()
            while child.poll() is None:
                elapsed=time.monotonic()-start
                if elapsed>=420:stop_owned('whole run420s deadline');break
                try:
                    sample=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True,timeout=min(10,max(0.01,420-elapsed)))
                    match=re.search(r'free percentage:\s*(\d+)%',sample.stdout)
                    record={'elapsedSeconds':elapsed,'exit':sample.returncode,'stdout':sample.stdout,'stderr':sample.stderr,'freePercent':int(match.group(1)) if match else None}
                    memory.write(json.dumps(record)+'\n');memory.flush()
                    if sample.returncode!=0 or not match or int(match.group(1))<30:stop_owned('memory below30% or unavailable');break
                except subprocess.TimeoutExpired:stop_owned('memory guard timeout');break
                elapsed=time.monotonic()-start
                if elapsed>=420:stop_owned('whole run420s deadline');break
                try:child.wait(timeout=min(2,max(0.01,420-elapsed)))
                except subprocess.TimeoutExpired:pass
            report['childExit']=child.wait();report['exit']=2 if 'watchdogAbort' in report else report['childExit'];report['wallSeconds']=time.monotonic()-start
    except Exception as error:
        stop_owned('wrapper failure: '+str(error));report['exit']=2;report['error']=str(error)
    finally:
        (run/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'receipt':str(run/'receipt.json'),'exit':report['exit']}))
    return report['exit']
if __name__=='__main__':raise SystemExit(main())
