Astra memory result — 2026-10-02, read-only

Select ONE bounded resource comparison with TURBO_QWEN_EXPERT_CACHE_RESIDENCY=0, retaining this serial probe, all other flags,1175 inputs,29-output budget,context8192,16 slots and30% floor. Timestamped evidence makes the optional residency request a specific intervention worth testing. It does not establish that residency caused all pressure or that disabling it will pass. Do not add an autorelease rewrite or other tuning to that comparison.

Reviewed run: run-20261002T192037.312697Z-2c4d23ceb7624e69b4d5bcfddf4da3d4. Parsed the three raw streams and supplied analysis. All11 current candidate files still match frozen-source-hashes.json, manifest SHA256538afbb9638ec79b36f65b6a9ea694cc41f36a55ba6252d2ccc1c6e2d159158a. No correctness/speed result completed.

Chronology, using aligned uptime clocks
Times below are relative to first wrapper memory sample. GiB means2^30 bytes. The process/system samples listed are nearest samples, not simultaneous measurements of each marker.

| Event | Marker seconds | Metal GiB | Nearby sample seconds | Footprint GiB | System compressor GiB | System wired GiB | Nearby free |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before model load |0.652|0.000|0.035|0.000|1.748|6.813|72%|
| After model load |3.840|4.559|4.040|4.711|9.462|7.280|46%|
| Layer20 done |84.070|6.934|84.036|6.734|14.104|7.224|31%|
| Layer38 done |155.903|8.623|156.039|8.444|13.917|7.168|32%|
| Residency request before |161.950|8.716|162.037|8.447|13.607|7.346|33%|
| Residency request returned |163.402|8.716|164.036|8.541|10.217|10.806|32%|
| Prefill done |165.788|8.689|166.040|8.521|9.289|12.417|30%|

The request interval is about1.452 s. The subsequent170-second sample has footprint8.527 GiB, compressor7.329 GiB, wired14.481 GiB and free30%. Free is NOT monotone:166s30%,168s34%,170s30%,172s28%. The watchdog then stops the child. There is a completed prefillDone marker but no decodeDone/diagnosticsStart. This excludes final diagnostic cloning before the last marker; it does not identify the precise instruction or establish which work was active at the stop.

What this supports
Metal allocation increases across prefill approximately in96 MiB expert-layer steps, ending near8.716 GiB (9.359 GB). This fits lazy allocation of the forty16-slot caches plus fixed model/state/scratch. It is not a sign of unbounded retained prompt matrices or completed commands. Allocated Metal size stays equal across the residency call, while system wired rises and compressor occupied pages fall afterward. The timing is consistent with changing physical residency rather than allocating duplicate expert buffers.
ModelExpertIO.swift register calls set.commit/requestResidency/queue.addResidencySet when the fortieth layer arrives, retaining the same1280 expert buffers,3.75 GiB total. Turning off that optional feature skips the residency owner/request. Original buffer ownership, cache slots, protected reads, source checks, exact routes/math, per-encoder resource use and GPU-lease settlement remain. This is a narrower causal intervention than changing memory accounting or object lifetimes.

What this does NOT support
Compression already grows from1.748 to9.462 GiB during the first four seconds, and reaches roughly14 GiB long before the request. Therefore the optional request cannot explain the entire resource barrier. Those are system compressor pages, with no owner attribution. They may include this child's compressed memory, displacement of other processes, or both. Do not label them external merely because the child's footprint is about8.5 GiB, and do not add/subtract footprint, Metal allocation and compressor bytes as disjoint pools. The public counters cannot partition that ownership.
The child lifetime maximum is9,212,255,600 bytes (~8.580 GiB), and highest live sampled footprint9,210,404,208 bytes. This is close to prior app-scale footprint and does not prove a child leak. It also does not make the system guard failure harmless or justify relaxing30%. Read bytes are process-attributed traffic, not unique working-set or physical SSD reads. request-return does not prove residency preparations completed by that instant.

After-exit sample correction
At172.006s proc_pid_rusage returns success with ri_phys_footprint=0 AND nonzero ri_proc_exit_abstime50888380367451. It is an exited-process record. Retain it as exit evidence and for its lifetime maximum, but exclude it from live footprint minima/end-minus-start calculations. The supplied analysis's first-to-last footprint change of−81992 bytes therefore is not a meaningful live-process memory trend. The nonzero resident-size residue in that same record is not a surviving live allocation measurement.

Fixed discriminator and stop rule
Run the same instrumented serial child once with only expert residency off, if Main authorizes and launch guards pass. Preserve phase/footprint/system samples and source/input hashes. Success for resource feasibility is completing all29 exact outputs and final state report above30%, with no residency request and the same40 caches/16 slots. It is not a speed gain or recovery qualification. If it again crosses30%, stop without another admission threshold or flag sweep; use the recorded prefill/decode phase and allocation/footprint trajectory to decide the next change. If it completes, any following causal serial/legacy/new comparison must use the same residency setting in all arms and disclose the changed condition.

Decision: test the existing residency switch off once as a controlled resource intervention. Earlier compressor growth remains an unresolved contributor, not a reason to relax the memory floor or rewrite allocation code.
