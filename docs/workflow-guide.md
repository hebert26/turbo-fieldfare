# Workflow guide

Use the global `/workflow` launcher from this project directory. Choose a project preset and give the work in a quoted `--task` value.

```text
/workflow --team turbo-high --task "Add the requested behavior to the importer"
```

`turbo-high` starts five agents for normal implementation:

- one Luna high implementation agent;
- two Luna xhigh agents for plan review and an independent initial review;
- two Terra medium agents for an independent initial review and verification.

Use it for work that needs implementation and review.

```text
/workflow --team smoke-test --task "Inspect the read-only workflow path"
```

`smoke-test` starts exactly three Luna low agents. They inspect files and complete the workflow events without changing files. Use it to exercise the read-only smoke workflow. Its result is a smoke-test outcome, not normal verification.

`--team` selects the preset. `--task` is required, and quote the task so it is passed as one value.

Project presets are defined in [`.pi/teams.yaml`](../.pi/teams.yaml). Agent profiles are in [`.pi/agents`](../.pi/agents/). The `/workflow` launcher is global.

`terra-general` and `prompt-reviewer` are agent profiles only. They do not create team presets.

After changing the launcher extension or a project agent profile, run `/reload` before starting a workflow.
