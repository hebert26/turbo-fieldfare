#!/usr/bin/env python3
"""Read saved decode receipts. Never run or control the model.

Conditional source inference: one routed command buffer per layer per decode
forward plus one extra buffer when a mixed hit/miss phase-1 split executes.
This does not estimate individual expert hits or distinguish all-hit/all-miss.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
EVIDENCE = Path('/Users/dev-machine/dev/VisionOS/Project-files/active/cache-layout-recovery/evidence')
SOURCES = {
    'journey27': EVIDENCE / 'gemma-journey27-20260907/model-trace.jsonl',
    'image_diagnostic': EVIDENCE / 'gemma-journey27-20260907/chat-image-diagnostic/model-trace-with-observation.jsonl',
    'journey28': Path('/tmp/gemma-journey28-20260907.jsonl'),
}
SNAPSHOT_NAMES = {
    'journey27': 'journey27.snapshot.jsonl',
    'image_diagnostic': 'image_diagnostic.snapshot.jsonl',
    'journey28': 'journey28_checkpoint.snapshot.jsonl',
}

rows, provenance = [], []
for name, path in SOURCES.items():
    snapshot = OUT.parent / 'journey-analysis' / SNAPSHOT_NAMES[name]
    data = snapshot.read_bytes()
    records, malformed = [], []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append((line_number, json.loads(line)))
        except json.JSONDecodeError:
            malformed.append(line_number)
    provenance.append(dict(name=name, path=str(path), snapshot_path=str(snapshot), bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(), malformed_lines=malformed,
        last_timestamp=max((r.get('timestamp_unix_seconds', 0) for _, r in records), default=0)))
    inputs = {r.get('step_id'): r.get('input', {}) for _, r in records if r.get('event') == 'input'}
    for line_number, record in records:
        if record.get('event') not in ('output', 'error', 'cancelled'):
            continue
        output = record.get('output', {})
        diagnostics = output.get('diagnostics', {})
        runner = diagnostics.get('runner', {})
        gpu = runner.get('gpu_completion_timing', {})
        input_row = inputs.get(record.get('step_id'), {})
        top_images = input_row.get('image_attachment_count', 0)
        result_images = sum(len(result.get('image_attachments', [])) for result in input_row.get('results', []))
        image_count = max(top_images, result_images)
        forwards = runner.get('decode_forward_count')
        base = 30 * forwards if isinstance(forwards, int) else None
        routed, shared, attention = (gpu.get(group, {}) for group in ('routed_experts', 'shared_experts', 'attention_router'))
        rc, sc, ac = (group.get('expected_buffer_count') for group in (routed, shared, attention))
        complete = bool(gpu) and all(group.get('expected_buffer_count', 0) > 0
            and group.get('expected_buffer_count') == group.get('valid_buffer_count')
            and group.get('milliseconds_per_forward') is not None for group in gpu.values())
        valid = record.get('event') == 'output' and base and complete and sc == base and ac == base and base <= rc <= 2 * base
        mixed = rc - base if valid else None
        remaining = base - mixed if valid else None
        row = dict(dataset=name, source_line=line_number, step_index=record.get('step_index'),
            step_id=record.get('step_id'), process_id=record.get('process_id'), event=record.get('event'),
            input_kind=input_row.get('kind'), image_count=image_count,
            prefill_chunk_tokens=input_row.get('settings', {}).get('prefill_chunk_tokens'),
            stop_reason=diagnostics.get('stop_reason'), decode_forwards=forwards,
            gpu_counts_complete=complete, shared_buffers=sc, attention_buffers=ac,
            routed_buffers=rc, layer_calls=base, bound_valid=bool(valid),
            mixed_split_calls=mixed, all_hit_upper_bound_calls=remaining,
            mixed_split_percent=100 * mixed / base if valid else None,
            all_hit_upper_bound_percent=100 * remaining / base if valid else None)
        rows.append(row)

groups = []
for name in SOURCES:
    for image_class in ('all', 'image', 'no_image'):
        chosen = [r for r in rows if r['dataset'] == name and r['bound_valid']
            and (image_class == 'all' or bool(r['image_count']) == (image_class == 'image'))]
        if not chosen:
            continue
        base = sum(r['layer_calls'] for r in chosen)
        mixed = sum(r['mixed_split_calls'] for r in chosen)
        groups.append(dict(dataset=name, image_class=image_class, rows=len(chosen),
            forwards=sum(r['decode_forwards'] for r in chosen), layer_calls=base,
            routed_buffers=sum(r['routed_buffers'] for r in chosen), mixed_split_calls=mixed,
            all_hit_upper_bound_calls=base-mixed, mixed_split_percent=100*mixed/base,
            all_hit_upper_bound_percent=100*(base-mixed)/base))

result = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
    caveat='Current-source interpretation; no complete build-source manifest. Split calls are a lower bound on mixed plans, all-hit minimum is zero. No per-layer expert identity, hit count, read-byte count, or scheduling latency is inferred.',
    provenance=provenance, aggregates=groups, rows=rows)
(OUT / 'cache-bounds.json').write_text(json.dumps(result, indent=2) + '\n')
with (OUT / 'cache-bounds.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(dict(provenance=provenance, aggregates=groups,
    rows_without_valid_bounds=[{k:r[k] for k in ('dataset','step_index','event','decode_forwards','gpu_counts_complete','shared_buffers','attention_buffers','routed_buffers')} for r in rows if not r['bound_valid']]), indent=2))
