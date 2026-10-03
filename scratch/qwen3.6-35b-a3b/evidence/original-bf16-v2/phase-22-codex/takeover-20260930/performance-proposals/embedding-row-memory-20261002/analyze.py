#!/usr/bin/env python3
"""Compare one saved candidate with the fixed completed serial baseline."""
import argparse
import datetime
import json
import pathlib
import sys
import uuid

BASE = pathlib.Path(__file__).resolve().parent
MEM = BASE.parent / 'baseline-memory-probe-20261002'
sys.path.insert(0, str(MEM))
import analyze as memory_checks

require = memory_checks.require
sha = memory_checks.sha
load_json = memory_checks.load_json
integer = memory_checks.integer
MATCH_FIELDS = ('outputIDs', 'acceptedRouteCount', 'acceptedRawRowCount',
                'acceptedRouteSHA256', 'acceptedRawRowSHA256', 'finalState')


def load_marker(run):
    errors = []
    rows = memory_checks.read_lines(run / 'probe.stderr', errors, phase_only=True)
    markers = [row for row in rows if row.get('phase') == 'afterModelLoad']
    return markers[0] if len(markers) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    run = args.run.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = args.output or run / ('analysis-' + stamp + '-' + uuid.uuid4().hex + '.json')
    verdict = {'schema': 1, 'runPath': str(run), 'validEvidence': False,
               'completeCorrectnessResult': False, 'baselineMatched': False,
               'speedClaim': False, 'exit': 2, 'errors': [],
               'limitations': ['One candidate uses an earlier saved baseline.',
                               'Timing and sampled memory differences are observations only.',
                               'Request timing excludes load. No app speed claim.',
                               'Allocated Metal bytes and process footprint are different measures.',
                               'The last saved phase does not prove the active phase at stop.']}
    try:
        control = load_json(BASE / 'control-manifest.json')
        receipt = load_json(run / 'receipt.json')
        verdict['runReceiptSHA256'] = sha(run / 'receipt.json')
        require(receipt['runPath'] == str(run) and receipt['requestedMode'] == 'serial', 'run identity differs')
        require(receipt['environment'] == control['environment'], 'candidate flags differ')
        require(receipt['controlSHA256'] == sha(BASE / 'control-manifest.json'), 'control changed')
        for name, wanted in control['reusedEvidence'].items():
            path = pathlib.Path(name)
            require(not path.is_absolute() and '..' not in path.parts, 'unsafe reused path')
            require(sha(MEM / path) == wanted, 'baseline or reused code changed: ' + name)
        raw = receipt['rawSHA256']
        require(isinstance(raw, dict) and 'probe.stderr' in raw and 'memory.jsonl' in raw,
                'missing raw evidence')
        for name, wanted in raw.items():
            path = pathlib.Path(name)
            require(not path.is_absolute() and '..' not in path.parts, 'unsafe raw path')
            require(sha(run / path) == wanted, 'raw evidence changed: ' + name)
        verdict['candidateMemory'] = memory_checks.memory_summary(run, receipt)
        require(receipt.get('integrityVerified') is True, 'source or binary proof did not complete')
        require(not verdict['candidateMemory']['phaseRecordErrors'], 'invalid phase records')
        baseline_run = MEM / control['baselineRun']
        old_marker, new_marker = load_marker(baseline_run), load_marker(run)
        old_bytes = old_marker.get('currentAllocatedSizeBytes') if old_marker else None
        new_bytes = new_marker.get('currentAllocatedSizeBytes') if new_marker else None
        reduction = old_bytes - new_bytes if type(old_bytes) is int and type(new_bytes) is int else None
        verdict['afterModelLoadMetal'] = {
            'baselineMarker': old_marker, 'candidateMarker': new_marker,
            'measuredReductionBytes': reduction,
            'staticEmbeddingTableBytes': 1017118720,
            'matchesStaticExpectedReduction': reduction == 1017118720,
            'missingMarkerOrCounter': reduction is None,
            'limit': 'This uses actual Metal markers. The removed table report field is a geometry assertion.'}
        verdict['validEvidence'] = True
        if receipt['exit'] != 0 or receipt.get('childExit') != 0 or receipt.get('watchdogAbort'):
            verdict['errors'].append('Candidate stopped. No complete comparison result.')
        else:
            require(sha(BASE / 'request.json') == memory_checks.REQUEST_SHA256, 'request changed')
            request = load_json(BASE / 'request.json')
            result = load_json(run / 'result.json')
            require(result == load_json(run / 'probe.stdout'), 'stdout and result differ')
            integer(receipt['launchMemoryFreePercent'], 'launch free memory', 50, 100)
            integer(receipt['observedMinimumFreePercent'], 'observed free memory', 30, 100)
            require(result['protectedEmbeddingRows'] is True, 'protected embedding path was not used')
            integer(result['protectedEmbeddingRowBytes'], 'embedding row bytes', 4096, 4096)
            integer(result['removedResidentEmbeddingBytes'], 'removed embedding bytes', 1017118720, 1017118720)
            integer(result['residentWeightBytes'], 'resident weight bytes', 1)
            verdict['candidateCorrectness'] = memory_checks.analyze_report(result, request)
            baseline_receipt = load_json(baseline_run / 'receipt.json')
            require(baseline_receipt['exit'] == baseline_receipt['childExit'] == 0
                    and baseline_receipt['integrityVerified'] is True
                    and not baseline_receipt.get('watchdogAbort'), 'baseline did not complete')
            baseline = load_json(baseline_run / 'result.json')
            require(baseline == load_json(baseline_run / 'probe.stdout'), 'baseline stdout and result differ')
            verdict['baselineCorrectness'] = memory_checks.analyze_report(baseline, request)
            verdict['matchedFields'] = {field: result[field] == baseline[field] for field in MATCH_FIELDS}
            require(all(verdict['matchedFields'].values()), 'candidate differs from saved exact baseline')
            verdict['baselineMemory'] = memory_checks.memory_summary(baseline_run, baseline_receipt)
            verdict['protectedEmbedding'] = {key: result[key] for key in (
                'protectedEmbeddingRows', 'protectedEmbeddingRowBytes',
                'residentWeightBytes', 'removedResidentEmbeddingBytes')}
            verdict['exploratoryTimingRatiosCandidateOverSavedBaseline'] = {
                key: result[key] / baseline[key] for key in (
                    'promptWallNanoseconds', 'decodeWallNanoseconds', 'requestWallNanoseconds')}
            verdict.update(completeCorrectnessResult=True, baselineMatched=True, exit=0)
    except (Exception, KeyboardInterrupt) as error:
        verdict['errors'].append(str(error))
    memory_checks.write_json(output, verdict)
    print(json.dumps({'receipt': str(output), 'validEvidence': verdict['validEvidence'],
                      'baselineMatched': verdict['baselineMatched'], 'exit': verdict['exit']}))
    return verdict['exit']


if __name__ == '__main__':
    raise SystemExit(main())
