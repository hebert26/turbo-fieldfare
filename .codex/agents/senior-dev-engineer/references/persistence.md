# Persistence Reference

Use this for SwiftData models, migrations, managers, contexts, relationships, queries, and saved state.

## Owned Areas

- `VisionCapture/Sources/Core/Models/`
- `VisionCapture/Sources/Core/Persistence/`
- SwiftData schema design;
- `@Model` relationships and delete rules;
- migrations and user-data preservation plans;
- manager APIs and `ModelContext` ownership;
- cross-context references through `PersistentIdentifier`;
- query shape, fetch limits, predicates, and prefetching.

## Working Rules

- Use the `swiftdata-relationships` skill for relationship changes.
- Keep model changes deliberate.
- Schema changes need a migration and clear data-preservation note.
- Never pass `@Model` instances across actors.
- Use Sendable DTOs or persistent identifiers across actor boundaries.
- Keep `ModelContext` inside the actor or manager that owns it.
- Do not fetch all rows and filter in Swift when a predicate can do the work.
- Add `fetchLimit` to any query that may grow.
- Save once per unit of work, not inside tight loops.
- Preserve the app-agnostic rule.

## Verification

Run `swift build` from `VisionCapture/`. Run focused persistence tests when model, manager, migration, or query
behaviour changes. Run broader `swift test` when schema or migration changes can affect many paths.

## Handoff Notes

Report:

- schema or persistence change made;
- actor/context ownership;
- migration impact;
- query and complexity notes;
- tests run and results;
- remaining data risks.
