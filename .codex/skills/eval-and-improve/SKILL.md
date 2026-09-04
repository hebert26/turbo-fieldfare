---
name: eval-and-improve
description: Evaluates code quality accuracy against enforcement standards, then iteratively fixes all violations until 100% compliance. Runs discover → evaluate → fix → re-evaluate loop. Use when user says "eval and improve", "quality loop", "fix all violations", "get to 100%".
---

# Eval and Improve

Autonomous loop that evaluates code quality, fixes violations, and re-evaluates until 100% accuracy.

## When to Use

- After major feature implementations to ensure quality
- When user wants to reach 100% code quality compliance
- Before releases or PR submissions
- When user says "eval and improve", "quality loop", "fix all violations"

## Process (RHEA Loop)

### Phase 1: Discover
Run pattern discovery to understand the codebase state.

```bash
# Run from project root
bash .codex/skills/code-quality-enforcement/scripts/discover.sh
```

### Phase 2: Evaluate
Run the full evaluation script to get a scored report.

```bash
# Run from project root
bash .codex/skills/eval-and-improve/scripts/evaluate.sh
```

This produces a scored report with:
- Total checks run
- Checks passed
- Checks failed (with file:line details)
- **Accuracy percentage** (target: 100%)

### Phase 3: Fix
For each violation found in Phase 2:

1. **Read the violating file** to understand context
2. **Determine the correct fix** based on the code-quality-enforcement SKILL.md rules
3. **Apply the fix** using Edit tool (surgical, minimal changes)
4. **Verify the fix** by re-running the specific check

#### Fix Priority Order:
1. **CRITICAL (Red)** - Fix first: architecture violations, crash risks, security issues
2. **WARNING (Orange)** - Fix second: memory leaks, concurrency issues
3. **QUALITY (Yellow)** - Fix third: logging, dead code, maintainability
4. **LEGACY (Purple)** - Fix last: pattern modernization

#### Fix Rules:
- **One violation at a time** - Don't batch unrelated fixes
- **Minimal changes only** - Fix the violation, don't refactor surroundings
- **Preserve behavior** - The fix must not change app functionality
- **Skip false positives** - Log them, don't "fix" correct code
- **Respect CLAUDE.md rules** - All fixes must follow project patterns

### Phase 4: Re-Evaluate
After fixing all violations in a category, re-run evaluation:

```bash
bash .codex/skills/eval-and-improve/scripts/evaluate.sh
```

Compare accuracy before and after. Repeat Phase 3-4 until 100%.

## Output

**Reports are saved to:**
```
<project-root>/enforcement-report/<YYYY-MM-DD-HHmm>-eval-improve-report.md
```

### Report Format:

```markdown
# Eval & Improve Report
**Date:** YYYY-MM-DD HH:mm
**Project:** <project-name>

## Summary
| Metric | Before | After |
|--------|--------|-------|
| Total Checks | N | N |
| Passed | N | N |
| Failed | N | 0 |
| Accuracy | X% | 100% |

## Iterations
### Iteration 1
- **Violations Found:** N
- **Violations Fixed:** N
- **False Positives:** N (with justification)
- **Accuracy:** X%

### Iteration 2
...

## Fixes Applied
| # | File | Line | Violation | Fix | Category |
|---|------|------|-----------|-----|----------|
| 1 | path | line | description | what was changed | CRITICAL/WARNING/QUALITY |

## False Positives (Excluded)
| # | File | Line | Detection | Reason for Exclusion |
|---|------|------|-----------|---------------------|

## Final Status
✅ 100% Accuracy Achieved (or ❌ with explanation)
```

## Iteration Limits

- **Max iterations:** 5
- **If stuck after 5 iterations:** Report remaining violations with analysis of why they can't be auto-fixed
- **Never force-fix:** If a "violation" is actually correct code, mark it as a false positive

## Category-Specific Fix Patterns

### Architecture Violations
```swift
// VIOLATION: Direct context access in View
@Environment(\.modelContext) var context
// FIX: Remove and use ViewModel via @Environment

// VIOLATION: context.fetch outside Manager
let items = try context.fetch(descriptor)
// FIX: Move to appropriate *Manager.swift, expose as method
```

### Concurrency Violations
```swift
// VIOLATION: ViewModel missing @MainActor
@Observable
class MyViewModel { }
// FIX: Add @MainActor above @Observable

// VIOLATION: .onAppear with Task
.onAppear { Task { await load() } }
// FIX: Replace with .task { await load() }

// VIOLATION: DispatchQueue.main.async
DispatchQueue.main.async { self.update() }
// FIX: await MainActor.run { update() }
```

### Crash Risk Violations
```swift
// VIOLATION: Force unwrap
let value = optional!
// FIX: guard let value = optional else { return }

// VIOLATION: try!
let data = try! JSONDecoder().decode(...)
// FIX: do { let data = try JSONDecoder()... } catch { logger.error(...) }
```

### Logging Violations
```swift
// VIOLATION: print() statement
print("Debug: \(value)")
// FIX: Logger.<category>.debug("Debug: \(value, privacy: .private)")

// VIOLATION: Inline Logger creation
let logger = Logger(subsystem: "com.app", category: "feature")
// FIX: Use Logger.<feature> from centralized extension
```

## Integration with Code Quality Enforcement

This skill depends on the `code-quality-enforcement` skill for:
- Validation rules and patterns
- Discovery script
- Reference documentation

Always read `.codex/skills/code-quality-enforcement/SKILL.md` for the authoritative list of checks.
