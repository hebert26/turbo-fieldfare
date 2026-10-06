#!/usr/bin/env python3
"""Compare two serial mode pairs. Keep every attempt and stop on failure."""
import argparse
import datetime
import json
import math
import os
import pathlib
import subprocess
import sys
import uuid

from evidence_helpers import BASE, sha
from analyze import analyze_report, load_json, integer
from run import FLAGS, HEAD, REQUEST_SHA256, require, write_json


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'invalid positive time: ' + label)


def validate_run(run_path, receipt, mode, expected, baseline, wrapper_hash):
    require(receipt['runPath'] == str(run_path), 'run path differs from receipt')
    integer(receipt['schema'], 'wrapper schema', minimum=1, maximum=1)
    integer(receipt['exit'], 'wrapper exit', maximum=0)
    integer(receipt['childExit'], 'child exit', maximum=0)
    require('error' not in receipt and 'watchdogAbort' not in receipt, 'wrapper aborted')
    require(receipt['requestedMode'] == mode, 'requested mode differs')
    require(receipt['commit'] == HEAD and receipt['requestSHA256'] == REQUEST_SHA256, 'pinned input differs')
    require(receipt['wrapperSHA256'] == wrapper_hash, 'wrapper changed')
    require(receipt['commitAfter'] == HEAD and receipt['binaryAfterSHA256'] == receipt['binarySHA256']
            and receipt['sourceHashesAfter'] == receipt['sourceHashes']
            and receipt['stageHashesAfter'] == receipt['stageHashes'], 'before/after identity differs')
    wanted_env = dict(FLAGS)
    require(receipt['environment'] == wanted_env, 'run flags differ')
    require(receipt['probeCommand'] == [str(BASE / 'build/release/TurboFieldfareCausalLookupDecode'),
            receipt['registrationPath'], str(BASE / 'request.json'), str(run_path / 'result.json'), mode],
            'actual CLI mode or command differs')
    positive(receipt['wallSeconds'], 'wrapper wall')
    require(receipt['wholeRunTimeoutSeconds'] == 420 and receipt['memorySamplingSeconds'] == 2,
            'wrapper guard settings differ')
    integer(receipt['minimumFreePercent'], 'wrapper minimum memory', minimum=30, maximum=100)
    integer(receipt['launchMemoryFreePercent'], 'launch free memory', minimum=30, maximum=100)
    preflight = pathlib.Path(receipt['preflightReceipt'])
    require(preflight.parent.resolve() == BASE and sha(preflight) == receipt['preflightReceiptSHA256'],
            'preflight evidence differs')
    pre = load_json(preflight)
    require(pre['passed'] is True and pre['exit'] == 0, 'preflight failed')
    require(isinstance(pre['checks'], list) and bool(pre['checks'])
            and all(check['passed'] is True for check in pre['checks']), 'preflight check failed')
    for key, name in [('stdout', 'probe.stdout'), ('stderr', 'probe.stderr'),
                      ('result', 'result.json'), ('memory', 'memory.jsonl')]:
        require(receipt[key + 'Path'] == str(run_path / name), 'evidence path differs: ' + key)
        require(sha(run_path / name) == receipt[key + 'SHA256'], 'evidence hash differs: ' + key)
    samples = [json.loads(line) for line in (run_path / 'memory.jsonl').read_text().splitlines()]
    require(bool(samples), 'wrapper memory samples missing')
    for sample in samples:
        require(sample['exit'] == 0, 'wrapper memory command failed')
        integer(sample['freePercent'], 'memory sample', minimum=30, maximum=100)
    require(min(sample['freePercent'] for sample in samples) == receipt['minimumFreePercent'],
            'wrapper memory minimum differs')
    result = load_json(run_path / 'result.json')
    require(load_json(run_path / 'probe.stdout') == result, 'stdout differs from result')
    require(type(result['schema']) is int and result['schema'] == 1, 'unknown report schema')
    require(result['mode'] == mode, 'actual report mode differs')
    summary = analyze_report(result, expected)
    if baseline is not None:
        old_receipt, old_result = baseline
        for key in ('binarySHA256', 'stageHashes', 'sourceHashes', 'stageHashesSHA256',
                    'sourceHashesSHA256', 'buildReceiptSHA256', 'requestSHA256', 'wrapperSHA256',
                    'registrationPath'):
            require(receipt[key] == old_receipt[key], 'run identity differs: ' + key)
        for key in ('outputIDs', 'finalState', 'acceptedRouteCount', 'acceptedRawRowCount',
                    'acceptedRouteSHA256', 'acceptedRawRowSHA256', 'sourceDescriptorSHA256'):
            require(result[key] == old_result[key], 'exact cross-run evidence differs: ' + key)
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registration')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    comparison = BASE / ('comparison-' + stamp + '-' + uuid.uuid4().hex)
    comparison.mkdir()
    report = {'schema': 1, 'comparisonPath': str(comparison), 'exit': 2, 'accepted': False,
              'outcome': 'invalid', 'speedClaimSupported': False,
              'modeOrder': [], 'runs': [], 'pairs': [], 'errors': [],
              'wrapperHashes': {'run.py': sha(BASE / 'run.py'), 'compare.py': sha(BASE / 'compare.py'),
                                'analyze.py': sha(BASE / 'analyze.py')},
              'gate': 'Advance only if each full decode ON/OFF ratio is <= 0.95 in both orders. '
                      'A first pair above 0.95 stops without a reverse run.',
              'headline': 'Whole decode includes all 29 sampled outputs and EOS; EOS remains unconsumed.',
              'limitations': ['Diagnostic schedule only. No app speed claim.',
                              'OFF uses one turn. ON uses per-cycle turns required for branch recovery.',
                              'Cycle, sampler and replay work stays in the whole decode measurement.',
                              'Golden assertion and final state readback occur after the timed loop.']}
    lock = BASE / '.compare.lock'
    lock_owned = False
    baseline = None
    try:
        with lock.open('x') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'comparisonPath': str(comparison)}) + '\n')
        lock_owned = True
        require(sha(BASE / 'request.json') == REQUEST_SHA256, 'frozen request differs')
        request = load_json(BASE / 'request.json')
        require(len(request['inputTokenIDs']) == 1175 and len(request['sampledTokenIDs']) == 29,
                'frozen input or output count differs')
        require(request['maximumContext'] == 8192, 'frozen context differs')
        require(request['sampledTokenIDs'][-1] == request['codecEOS'], 'frozen request lacks final EOS')
        for order in [('off', 'on'), ('on', 'off')]:
            pair_results = {}
            pair_paths = {}
            for mode in order:
                command = [sys.executable, str(BASE / 'run.py'), args.registration, '--mode', mode]
                attempt = {'mode': mode, 'command': command, 'exit': None,
                           'validated': False, 'errors': []}
                report['modeOrder'].append(mode)
                report['runs'].append(attempt)
                label = 'run-' + str(len(report['runs'])) + '-' + mode
                attempt.update(wrapperStdoutPath=str(comparison / (label + '.stdout')),
                               wrapperStderrPath=str(comparison / (label + '.stderr')))
                try:
                    value = subprocess.run(command, cwd=BASE, capture_output=True, text=True)
                    attempt['exit'] = value.returncode
                    (comparison / (label + '.stdout')).write_text(value.stdout)
                    (comparison / (label + '.stderr')).write_text(value.stderr)
                    summary = json.loads(value.stdout)
                    run_path = pathlib.Path(summary['runPath'])
                    require(run_path.is_absolute() and run_path.parent.resolve() == BASE
                            and run_path.name.startswith('run-'), 'invalid returned run path')
                    require(summary['receipt'] == str(run_path / 'receipt.json'), 'returned receipt path differs')
                    attempt.update(runPath=str(run_path), receiptPath=summary['receipt'])
                    receipt = load_json(run_path / 'receipt.json')
                    attempt.update(receipt=receipt, receiptSHA256=sha(run_path / 'receipt.json'))
                    require(value.returncode == 0 and summary['exit'] == 0, 'fresh wrapper run failed')
                    result, analysis = validate_run(run_path, receipt, mode, request,
                                          baseline, report['wrapperHashes']['run.py'])
                    attempt.update(validated=True, analysis=analysis,
                                   promptWallNanoseconds=result['promptWallNanoseconds'],
                                   decodeWallNanoseconds=result['decodeWallNanoseconds'],
                                   wallSeconds=receipt['wallSeconds'], outputCount=len(result['outputIDs']),
                                   internalMinimumFreePercent=result['minimumFreePercent'],
                                   wrapperMinimumFreePercent=receipt['minimumFreePercent'],
                                   binarySHA256=receipt['binarySHA256'],
                                   sourceHashesSHA256=receipt['sourceHashesSHA256'],
                                   timingLimits=result['timingLimits'])
                    write_json(comparison / (label + '-analysis.json'), analysis)
                    if baseline is None:
                        baseline = (receipt, result)
                    pair_results[mode] = result
                    pair_paths[mode] = str(run_path)
                except (Exception, KeyboardInterrupt) as error:
                    attempt['errors'].append(str(error))
                    raise
                finally:
                    write_json(comparison / (label + '-verdict.json'), attempt)
            on = pair_results['on']['decodeWallNanoseconds']
            off = pair_results['off']['decodeWallNanoseconds']
            ratio = on / off
            pair = {'modeOrder': list(order), 'runPaths': pair_paths,
                    'onDecodeWallNanoseconds': on, 'offDecodeWallNanoseconds': off,
                    'fullDecodeOnOffRatio': ratio, 'passed': on * 100 <= off * 95,
                    'notSlower': on <= off,
                    'onOutputsIncludingEOSPerSecond': 29 * 1e9 / on,
                    'offOutputsIncludingEOSPerSecond': 29 * 1e9 / off}
            report['pairs'].append(pair)
            if not pair['passed']:
                report['outcome'] = 'slower' if on > off else 'no_speed_claim'
                raise ValueError('full decode ON/OFF ratio exceeds 0.95 for ' + '/'.join(order))
        require(len(report['pairs']) == 2 and len(report['runs']) == 4, 'both mode orders required')
        report.update(accepted=True, exit=0, outcome='advance', speedClaimSupported=True)
    except (Exception, KeyboardInterrupt) as error:
        report['errors'].append(str(error))
    finally:
        if lock_owned:
            lock.unlink()
        write_json(comparison / 'receipt.json', report)
        print(json.dumps({'comparisonPath': str(comparison), 'receipt': str(comparison / 'receipt.json'),
                          'accepted': report['accepted'], 'exit': report['exit']}, allow_nan=False))
    return report['exit']


if __name__ == '__main__':
    raise SystemExit(main())
