# App-Agnostic Reference

Use this whenever code, docs, tests, prompts, workflows, or examples could depend on one known app.

VisionCapture is an enterprise automation product. Customers run it against apps this team has never seen.

## Hard Rules

- All features must work for any iOS app running in the Simulator.
- Never hardcode customer app labels, bundle ids, screen names, navigation paths, or text strings as product logic.
- Classify UI elements by standard accessibility roles such as `tab`, `cell`, `button`, `staticText`, and similar
  generic roles.
- Build navigation trees, step libraries, and workflows from generic accessibility data.
- Treat visible text as observed data, not routing policy.
- In docs, tests, prompts, and examples, prefer placeholders such as `<your.app.bundle.id>` and `[Screen title]`.
- Do not use NestMind or any other real app as a baked-in assumption.

## Audit Targets

Check for:

- hardcoded app labels, screen names, button names, and bundle ids;
- navigation rules based on visible text instead of accessibility roles and structure;
- LLM prompts that steer toward a specific app;
- tests or fixtures leaking into product logic;
- provider-specific behaviour presented as generic automation behaviour;
- docs that teach users to copy app-specific examples as product rules.

## Acceptable Uses Of Labels

Labels can be used when they are:

- observed data in a captured accessibility tree;
- user-provided input for a specific run;
- test fixture values isolated from product logic;
- examples clearly marked with placeholders.

## Fix Direction

Replace app-specific assumptions with:

- accessibility role and hierarchy checks;
- relative structure;
- stable element metadata;
- generic screen/state descriptors;
- user-provided selectors or workflow data;
- Core policies that can work across many apps.
