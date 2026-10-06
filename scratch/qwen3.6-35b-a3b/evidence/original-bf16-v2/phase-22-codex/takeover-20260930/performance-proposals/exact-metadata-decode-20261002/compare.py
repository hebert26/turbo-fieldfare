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
from run import FLAGS, HEAD, REQUEST_SHA256, require, write_json

CASE_KEYS = [(position, layer) for position in range(1175, 1203) for layer in range(40)]
CPU_FIELDS = ('processCPUUserMicroseconds', 'processCPUSystemMicroseconds',
              'decodeCPUUserMicroseconds', 'decodeCPUSystemMicroseconds')


def load_json(path):
    def reject(value):
        raise ValueError('non-finite JSON value: ' + value)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    return json.loads(path.read_text(), parse_constant=reject, object_pairs_hook=unique)


def integer(value, label, minimum=0, maximum=2**64 - 1):
    require(type(value) is int and minimum <= value <= maximum, 'invalid integer: ' + label)


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'invalid positive time: ' + label)


def integer_list(value, label, maximum=2**63 - 1, nullable=False):
    require(isinstance(value, list), 'invalid list: ' + label)
    for item in value:
        if not (nullable and item is None):
            integer(item, label, maximum=maximum)


def validate_rows(rows, label):
    require(isinstance(rows, list) and len(rows) == 1120, label + ' must have 1120 rows')
    keys = []
    for row in rows:
        require(isinstance(row, dict), 'invalid ' + label + ' row')
        integer(row['position'], label + ' position')
        integer(row['layer'], label + ' layer')
        keys.append((row['position'], row['layer']))
        integer_list(row['experts'], label + ' experts')
        require(bool(row['experts']) and len(set(row['experts'])) == len(row['experts']),
                label + ' expert list empty or duplicated')
        if label == 'routes':
            integer_list(row['weightBits'], 'route weight bits', maximum=2**32 - 1)
            require(len(row['weightBits']) == len(row['experts']), 'route weight count differs')
        else:
            integer_list(row['assignedSlots'], 'assigned slots')
            integer_list(row['missIndices'], 'miss indices', maximum=len(row['experts']) - 1)
            integer_list(row['residents'], 'residents', nullable=True)
            require(len(row['assignedSlots']) == len(row['experts']), 'assigned slot count differs')
            require(len(set(row['assignedSlots'])) == len(row['assignedSlots']), 'duplicate assigned slots')
            require(len(set(row['missIndices'])) == len(row['missIndices']), 'duplicate miss indices')
    require(keys == CASE_KEYS, label + ' cases missing, duplicated, or unsorted')


def validate_run(run_path, receipt, mode, expected, baseline, wrapper_hash):
    require(receipt['runPath'] == str(run_path), 'run path differs from receipt')
    integer(receipt['schema'], 'wrapper schema', minimum=1, maximum=1)
    integer(receipt['exit'], 'wrapper exit', maximum=0)
    integer(receipt['childExit'], 'child exit', maximum=0)
    require('error' not in receipt and 'watchdogAbort' not in receipt, 'wrapper aborted')
    require(receipt['requestedMode'] == mode, 'requested mode differs')
    require(receipt['commit'] == HEAD and receipt['requestSHA256'] == REQUEST_SHA256, 'pinned input differs')
    require(receipt['wrapperSHA256'] == wrapper_hash, 'wrapper changed')
    wanted_env = dict(FLAGS, TURBO_QWEN_DECODE_METADATA_FOUR='1' if mode == 'on' else '0')
    require(receipt['environment'] == wanted_env, 'run flags differ')
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
    integer_list(result['outputIDs'], 'output IDs')
    require(result['outputIDs'] == expected and len(result['outputIDs']) == 29,
            'all 29 exact output IDs, including EOS, required')
    require(result['expectedOutputIDsMatched'] is True, 'reported output mismatch')
    integer(result['promptWallNanoseconds'], 'prompt wall', minimum=1)
    integer(result['decodeWallNanoseconds'], 'full decode wall', minimum=1)
    require(isinstance(result['forwardWallNanoseconds'], list)
            and len(result['forwardWallNanoseconds']) == 28, '28 consumed forward times required')
    for value in result['forwardWallNanoseconds']:
        integer(value, 'forward wall', minimum=1)
    validate_rows(result['routes'], 'routes')
    validate_rows(result['plans'], 'plans')
    require(all(route['experts'] == plan['experts']
                for route, plan in zip(result['routes'], result['plans'])), 'route/plan experts differ')
    safety = result['safety']
    require(all(safety[key] is True for key in ('captureComplete', 'runnerUsable')),
            'capture or runner safety failed')
    integer(safety['position'], 'final position')
    integer(safety['pendingTokens'], 'pending tokens')
    require(safety['position'] == 1203 and safety['pendingTokens'] == 1, 'final state differs')
    integer(result['minimumFreePercent'], 'internal minimum memory', minimum=30, maximum=100)
    source = result['sourceValidation']
    require(isinstance(source, dict), 'source validation evidence missing')
    for key in ('count', 'wallNanoseconds', 'failures', 'parallelCount',
                'parallelWallNanoseconds', 'entryChecks', 'maximumEntryFDs'):
        integer(source[key], 'source validation ' + key)
    require(source['count'] > 0 and source['entryChecks'] > 0, 'source validation evidence empty')
    require(source['failures'] == 0, 'source validation failed')
    require(source['wallNanoseconds'] > 0, 'source validation wall time empty')
    require(source['parallelCount'] <= source['count'], 'parallel count exceeds validation count')
    integer(source['maximumEntryFDs'], 'measured entry file count', minimum=1,
            maximum=4 if mode == 'on' else 1)
    if mode == 'on':
        require(source['parallelCount'] > 0 and source['parallelWallNanoseconds'] > 0,
                'parallel validation evidence empty')
        require(source['parallelWallNanoseconds'] <= source['parallelCount'] * 1050000,
                'ON parallel validation mean exceeds 1050000ns')
    else:
        require(source['parallelCount'] == 0 and source['parallelWallNanoseconds'] == 0,
                'OFF used parallel validation')
    for key in CPU_FIELDS:
        integer(result[key], key)
    require(result['processCPUUserMicroseconds'] >= result['decodeCPUUserMicroseconds']
            and result['processCPUSystemMicroseconds'] >= result['decodeCPUSystemMicroseconds'],
            'decode CPU usage exceeds whole process usage')
    if baseline is not None:
        old_receipt, old_result = baseline
        for key in ('binarySHA256', 'stageHashes', 'sourceHashes', 'stageHashesSHA256',
                    'sourceHashesSHA256', 'buildReceiptSHA256', 'requestSHA256', 'wrapperSHA256',
                    'registrationPath'):
            require(receipt[key] == old_receipt[key], 'run identity differs: ' + key)
        for key in ('outputIDs', 'routes', 'plans', 'safety'):
            require(result[key] == old_result[key], 'exact cross-run evidence differs: ' + key)
        for key in ('count', 'entryChecks'):
            require(source[key] == old_result['sourceValidation'][key],
                    'source validation count differs: ' + key)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registration')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    comparison = BASE / ('comparison-' + stamp + '-' + uuid.uuid4().hex)
    comparison.mkdir()
    report = {'schema': 1, 'comparisonPath': str(comparison), 'exit': 2, 'accepted': False,
              'modeOrder': [], 'runs': [], 'pairs': [], 'errors': [],
              'wrapperHashes': {'run.py': sha(BASE / 'run.py'), 'compare.py': sha(BASE / 'compare.py')},
              'gate': 'Each full decode ON/OFF ratio must be <= 0.97 in both orders; '
                      'each ON parallel validation mean must be <= 1050000ns.',
              'headline': 'Full decode includes all 29 exact outputs and EOS; 28 forwards consume tokens.',
              'limitations': ['Diagnostic replay only. No app speed claim.',
                              'Validation means and forward groups do not replace the full decode gate.']}
    lock = BASE / '.compare.lock'
    lock_owned = False
    baseline = None
    try:
        with lock.open('x') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'comparisonPath': str(comparison)}) + '\n')
        lock_owned = True
        require(sha(BASE / 'request.json') == REQUEST_SHA256, 'frozen request differs')
        request = load_json(BASE / 'request.json')
        integer_list(request['inputTokenIDs'], 'input IDs')
        integer_list(request['sampledTokenIDs'], 'expected IDs')
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
                    result = validate_run(run_path, receipt, mode, request['sampledTokenIDs'],
                                          baseline, report['wrapperHashes']['run.py'])
                    attempt.update(validated=True, promptWallNanoseconds=result['promptWallNanoseconds'],
                                   decodeWallNanoseconds=result['decodeWallNanoseconds'],
                                   wallSeconds=receipt['wallSeconds'], outputCount=len(result['outputIDs']),
                                   consumedForwardCount=len(result['forwardWallNanoseconds']),
                                   sourceValidation=result['sourceValidation'],
                                   parallelValidationMeanNanoseconds=(
                                       result['sourceValidation']['parallelWallNanoseconds']
                                       / result['sourceValidation']['parallelCount'] if mode == 'on' else None),
                                   cpuUsageMicroseconds={key: result[key] for key in CPU_FIELDS},
                                   routeCount=len(result['routes']),
                                   planCount=len(result['plans']), safety=result['safety'],
                                   internalMinimumFreePercent=result['minimumFreePercent'],
                                   wrapperMinimumFreePercent=receipt['minimumFreePercent'],
                                   binarySHA256=receipt['binarySHA256'],
                                   sourceHashesSHA256=receipt['sourceHashesSHA256'],
                                   secondaryForwardGroupsNanoseconds={
                                       'first3': sum(result['forwardWallNanoseconds'][:3]),
                                       'later25': sum(result['forwardWallNanoseconds'][3:])})
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
                    'fullDecodeOnOffRatio': ratio, 'passed': on * 100 <= off * 97,
                    'onParallelValidationMeanNanoseconds': (
                        pair_results['on']['sourceValidation']['parallelWallNanoseconds']
                        / pair_results['on']['sourceValidation']['parallelCount']),
                    'onMaximumEntryFDs': pair_results['on']['sourceValidation']['maximumEntryFDs'],
                    'offMaximumEntryFDs': pair_results['off']['sourceValidation']['maximumEntryFDs'],
                    'sourceValidationCount': pair_results['off']['sourceValidation']['count'],
                    'sourceEntryChecks': pair_results['off']['sourceValidation']['entryChecks'],
                    'onOutputsIncludingEOSPerSecond': 29 * 1e9 / on,
                    'offOutputsIncludingEOSPerSecond': 29 * 1e9 / off}
            report['pairs'].append(pair)
            require(pair['passed'], 'full decode ON/OFF ratio exceeds 0.97 for ' + '/'.join(order))
        require(len(report['pairs']) == 2 and len(report['runs']) == 4, 'both mode orders required')
        report.update(accepted=True, exit=0)
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
