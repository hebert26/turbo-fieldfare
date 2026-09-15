# Working with Hebert

Use short, plain paragraphs. State each fact once. Put the key outcome or next action last. Use lists only when they help scanning. Avoid jargon, decorative headings, canned wording, and analogies. Explain technical terms when needed.

Complete authorised work without repeated permission requests. Preserve unrelated edits and stay in scope. Verify claims proportionately. Open browser-viewed files in Safari. Unless the user explicitly directs another path, `/Users/dev-machine/Documents/Idea Home/turboCharge/` is only for documentation and plans: planning documents go under `Project-files/active/`, while HTML documents, including HTML plans, go under `Project-files/human/`. All other artifacts, including weights, data, downloads, scripts, build outputs, and runtime logs, stay in appropriate repository-owned locations. Existing source and repository files stay where they are.

Delegate only bounded, useful work. Give the owner scope, paths, completion evidence, and exclusive write ownership. Review delegated changes and evidence before reporting them. Run one model process at a time.

# TurboFieldfare

This checkout runs and reports existing Gemma 4 inference behavior. Do not edit source, change defaults, or optimise unless requested. `Sources/TurboFieldfareFormat/` owns the `.gturbo` v1 contract; runtime, repacker, CLI, server, and app modules are named for their targets. Use `Scripts/test.sh` for package tests. Read `README.md` for current commands and feature behavior.

Before any model run, require macOS 26+, Swift 6.2+, sufficient disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.gturbo`, and no process matching `TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm`. If a check fails, report it and stop; never kill apps, download or duplicate the model, create a worktree, purge caches, or reinstall it. Run one app, CLI, or model-using test at once.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS. Do not expose, proxy, or tunnel it. Start no second model process and stop only a server you started. For benchmarks, use the repository’s community benchmark guidance without experimental controls or profiling. Report commit, hardware/RAM, macOS, Swift, command, exit code, complete timing footer or error, and deviations.

Images need a valid adjacent `.vision.gturbo` pack. If it is missing or invalid, say image support is unavailable; never silently ignore an accepted image.
