# Workflow presets

Use a project-local preset with:

```text
/workflow --team <team-name> --task "Describe the work"
```

The command reads only `.pi/teams.yaml` and `.pi/agents/` in the active project. It rejects missing tasks, unknown presets, global profiles, incompatible model and thinking settings, and profiles missing required team workflow tools before creating a roster or starting an agent.

`teams.yaml` requires `version: 1` and a `teams:` mapping. Each preset has `mode: implementation`, `mode: read-only`, or `mode: smoke-test`, plus `members:`. Each member starts with `- agent: <project-profile-name>` and has `duties: [<role>]`. Roles are `implementation`, `plan_review`, `initial_review_luna`, `initial_review_terra`, `verification`, and `escalation`.

This project provides:

```text
/workflow --team turbo-high --task "Describe the implementation work"
/workflow --team smoke-test --task "Inspect this read-only workflow path"
```

`turbo-high` uses the local implementation, Luna xhigh, Terra medium, and verification profiles. `smoke-test` launches three Luna low agents and reports a smoke-test outcome only.
