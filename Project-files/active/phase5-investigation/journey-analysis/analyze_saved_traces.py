"""Read saved traces and freeze complete live records. Does not touch a runtime."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import csv
import hashlib
import json
import statistics

OUT = Path(__file__).resolve().parent
BASE = Path('/Users/dev-machine/dev/VisionOS/Project-files/active/cache-layout-recovery/evidence')
SOURCES = {
    'journey27': BASE / 'gemma-journey27-20260907/model-trace.jsonl',
    'image_diagnostic': BASE / 'gemma-journey27-20260907/chat-image-diagnostic/model-trace-with-observation.jsonl',
    'journey28_checkpoint': Path('/tmp/gemma-journey28-20260907.jsonl'),
}


def result_objects(record):
    values = []
    for result in record.get('input', {}).get('results', []):
        try:
            value, _ = json.JSONDecoder().raw_decode(result.get('content', ''))
            if isinstance(value, dict):
                values.append(value)
        except ValueError:
            pass
    return values


def aggregate(rows):
    if not rows:
        return {'count': 0}
    tokens = sum(r['generated_tokens'] or 0 for r in rows)
    decode = sum(r['decode_seconds'] or 0 for r in rows)
    forward_count = sum(r['decode_forward_count'] or 0 for r in rows)
    totals = {key: (sum(r[key] or 0 for r in rows) if any(r[key] is not None for r in rows) else None) for key in (
        'prefill_seconds', 'elapsed_seconds', 'input_to_output_seconds',
        'computed_prefill_tokens', 'cached_prompt_tokens', 'thinking_tokens',
        'unknown_hidden_channel_tokens', 'tool_call_tokens', 'visible_response_tokens', 'next_input_gap_seconds')}
    answer = {'count': len(rows), 'generated_tokens': tokens,
              'decode_seconds': decode, 'weighted_tokens_per_second': tokens / decode,
              'median_tokens_per_second': statistics.median(r['tokens_per_second'] for r in rows),
              'decode_forward_count': forward_count, **totals}
    gaps = [r['next_input_gap_seconds'] for r in rows if r['next_input_gap_seconds'] is not None]
    answer['next_input_gap_count'] = len(gaps)
    if gaps:
        answer['next_input_gap_median_seconds'] = statistics.median(gaps)
        answer['next_input_gap_max_seconds'] = max(gaps)
    if forward_count:
        timing_keys = [k for k in rows[0] if k.endswith('_milliseconds_per_forward')]
        answer['forward_weighted_timings'] = {
            k: sum((r.get(k) or 0) * (r['decode_forward_count'] or 0) for r in rows) / forward_count
            for k in timing_keys}
    return answer


all_summary = {}
summary_file = OUT / 'trace-summary.json'
previous = json.loads(summary_file.read_text()) if summary_file.exists() else {}
for name, path in SOURCES.items():
    snapshot = OUT / (name + '.snapshot.jsonl')
    # Existing snapshots are the stable replay inputs. Never replace them on rerun.
    captured = previous.get(name, {}).get('captured_utc', datetime.now(timezone.utc).isoformat())
    raw = snapshot.read_bytes() if snapshot.exists() else path.read_bytes()
    complete = []
    records = []
    excluded = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            record = json.loads(line)
            if not line.endswith(b'\n'):
                excluded.append(line_number)
                continue
        except ValueError:
            excluded.append(line_number)
            continue
        complete.append(line)
        record['_line'] = line_number
        records.append(record)
    frozen = b''.join(complete)
    if not snapshot.exists():
        snapshot.write_bytes(frozen)
    inputs = {r['step_id']: r for r in records if r['event'] == 'input'}
    rows = []
    for index, record in enumerate(records):
        if record['event'] not in ('output', 'error'):
            continue
        output = record['output']
        diagnostics = output.get('diagnostics') or {}
        runner = diagnostics.get('runner') or {}
        gpu = runner.get('gpu_completion_timing') or {}
        inp = inputs.get(record['step_id'])
        next_inp = next((r for r in records[index+1:] if r['event'] == 'input'), None)
        # Gaps across separate human requests are deliberately not tool latency.
        same_turn = (next_inp and next_inp['conversation_id'] == record['conversation_id']
                     and next_inp['turn_index'] == record['turn_index'])
        result_values = result_objects(inp or {})
        progress = output.get('structured_progress') or {}
        row = {
            'source_line': record['_line'], 'input_line': inp['_line'] if inp else None,
            'step_index': record['step_index'], 'turn_index': record['turn_index'],
            'event': record['event'], 'timestamp_unix_seconds': record['timestamp_unix_seconds'],
            'input_kind': inp['input'].get('kind') if inp else None,
            'input_operation': '|'.join(str(v.get('operation', '')) for v in result_values),
            'input_outcome': '|'.join(str(v.get('outcome', '')) for v in result_values),
            'input_observation_outcome': '|'.join(str(v.get('observation_outcome', '')) for v in result_values),
            'input_code': '|'.join(str(v.get('code', '')) for v in result_values),
            'image_attachment_count': inp['input'].get('image_attachment_count', 0) if inp else None,
            'output_action': '|'.join(c.get('arguments', {}).get('action', '') for c in output.get('tool_calls', [])),
            'input_to_output_seconds': record['timestamp_unix_seconds'] - inp['timestamp_unix_seconds'] if inp else None,
            'next_input_gap_seconds': next_inp['timestamp_unix_seconds'] - record['timestamp_unix_seconds'] if same_turn else None,
            'next_input_line': next_inp['_line'] if same_turn else None,
            'elapsed_seconds': output.get('elapsed_seconds'),
            'decode_forward_count': runner.get('decode_forward_count'),
        }
        for key in ('generated_tokens', 'decode_seconds', 'prefill_seconds', 'tokens_per_second',
                    'computed_prefill_tokens', 'cached_prompt_tokens', 'conversation_tokens',
                    'prompt_tokens', 'time_to_first_token_seconds', 'stop_reason'):
            row[key] = diagnostics.get(key)
        for key in ('thinking_tokens', 'unknown_hidden_channel_tokens', 'tool_call_tokens', 'visible_response_tokens', 'channel_label_tokens'):
            row[key] = progress.get(key)
        for key in ('router_wait', 'io', 'cb1', 'cb2', 'head'):
            full_key = key + '_milliseconds_per_forward'
            row[full_key] = runner.get(full_key)
        for key in ('attention_router', 'full_attention_router', 'sliding_attention_router', 'shared_experts', 'routed_experts'):
            value = gpu.get(key, {})
            row['gpu_' + key + '_milliseconds_per_forward'] = value.get('milliseconds_per_forward')
            row['gpu_' + key + '_valid_buffer_count'] = value.get('valid_buffer_count')
            row['gpu_' + key + '_expected_buffer_count'] = value.get('expected_buffer_count')
        rows.append(row)
    with (OUT / (name + '.steps.csv')).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = [r for r in rows if r['event'] == 'output']
    errors = [r for r in rows if r['event'] == 'error']
    grouping = {}
    for field in ('output_action', 'input_operation', 'input_outcome'):
        group = defaultdict(list)
        for row in completed:
            group[row[field]].append(row)
        grouping[field] = {key: aggregate(values) for key, values in group.items()}
    context_groups = defaultdict(list)
    for row in completed:
        n = row['conversation_tokens']
        context_groups['0-8191' if n < 8192 else '8192-32767' if n < 32768 else '32768-49151' if n < 49152 else '49152+'].append(row)
    first_input = next(r for r in records if r['event'] == 'input')
    all_summary[name] = {
        'source_path': str(path), 'captured_utc': captured,
        'source_read_sha256': hashlib.sha256(raw).hexdigest(), 'source_read_bytes': len(raw),
        'snapshot_path': str(snapshot), 'snapshot_sha256': hashlib.sha256(frozen).hexdigest(),
        'snapshot_bytes': len(frozen), 'complete_lines': len(records), 'excluded_source_lines': excluded,
        'first_timestamp': records[0]['timestamp_unix_seconds'],
        'last_timestamp': records[-1]['timestamp_unix_seconds'],
        'last_event': records[-1]['event'], 'last_step': records[-1]['step_index'],
        'trace_span_seconds': records[-1]['timestamp_unix_seconds'] - records[0]['timestamp_unix_seconds'],
        'event_counts': dict(Counter(r['event'] for r in records)),
        'settings': first_input['input']['settings'],
        'process_ids': sorted(set(r['process_id'] for r in records)),
        'model_directory': first_input['input'].get('model_directory'),
        'completed': aggregate(completed), 'errors': aggregate(errors),
        'max_completed_context': max(r['conversation_tokens'] for r in completed),
        'image_input_count': sum(r['input'].get('image_attachment_count', 0) for r in records if r['event']=='input'),
        'grouping': grouping, 'context_groups': {key: aggregate(values) for key, values in context_groups.items()},
        'gpu_invalid_rows': [r['source_line'] for r in completed if any(
            r['gpu_' + key + '_valid_buffer_count'] != r['gpu_' + key + '_expected_buffer_count']
            for key in ('attention_router', 'full_attention_router', 'sliding_attention_router', 'shared_experts', 'routed_experts'))],
    }

(OUT / 'trace-summary.json').write_text(json.dumps(all_summary, indent=2) + '\n')
for name, summary in all_summary.items():
    print(json.dumps({k:v for k,v in summary.items() if k not in ('grouping','settings','context_groups')}, indent=2))
