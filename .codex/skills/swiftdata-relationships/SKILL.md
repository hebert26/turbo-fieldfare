---
name: swiftdata-relationships
description: Defines the correct SwiftData @Relationship patterns used in VisionCapture. Use when creating, modifying, or working with model relationships. Prevents common mistakes like incorrect inverse relationships, wrong delete rules, and missing @Relationship annotations.
---

# SwiftData @Relationship Skill

## Purpose
This skill ensures correct SwiftData @Relationship usage following VisionCapture's established patterns. SwiftData relationships are **IMMUTABLE in production** — once set, they cannot be changed without breaking the persisted store and the user's captured workflows.

## Core Principle: Parent Owns Children
**Pattern**: Parent has `@Relationship` with children, children have a simple `var` reference to parent.

## Relationship Patterns (From VisionCapture Codebase)

### Pattern 1: One-to-Many with Cascade Delete
**Example**: `Workflow` → `WorkflowStep` (a Workflow has many WorkflowSteps)

**Parent (`Sources/Core/Models/Workflow.swift`)**:
```swift
// Relationship: Workflow has many WorkflowSteps
// Delete Rule: .cascade
// Explanation: When a Workflow is deleted, all its WorkflowSteps are also deleted.
@Relationship(deleteRule: .cascade, inverse: \WorkflowStep.workflow)
var steps: [WorkflowStep]
```

**Child (`Sources/Core/Models/WorkflowStep.swift`)**:
```swift
// Relationship: WorkflowStep belongs to one Workflow
// Delete Rule: .nullify (default on the child side)
var workflow: Workflow?
```

**Key Points**:
- ✅ Parent uses `@Relationship` with `deleteRule: .cascade` and `inverse:`
- ✅ Child uses plain `var` (no `@Relationship` annotation)
- ✅ Array type for parent (`[WorkflowStep]`), optional single type for child (`Workflow?`)
- ✅ Always document with comments explaining the relationship and delete behavior

### Pattern 2: One-to-Many, Same Parent, Multiple Children
**Example**: `Workflow` → `WorkflowParameter` (a Workflow has many parameters alongside its steps)

```swift
@Model
final class Workflow {
    @Relationship(deleteRule: .cascade, inverse: \WorkflowStep.workflow)
    var steps: [WorkflowStep]

    @Relationship(deleteRule: .cascade, inverse: \WorkflowParameter.workflow)
    var parameters: [WorkflowParameter]
}
```

Both `WorkflowStep` and `WorkflowParameter` carry a plain `var workflow: Workflow?` on the child side.

### Pattern 3: Session → Captures (Timeline)
**Example**: `CaptureSession` → `Capture` (`Sources/Core/Models/Models.swift`)

```swift
@Model
final class CaptureSession {
    // Relationship: CaptureSession has many Captures
    // Delete Rule: .cascade
    // Explanation: When a CaptureSession is deleted, all its Captures are also deleted.
    @Relationship(deleteRule: .cascade, inverse: \Capture.session)
    var captures: [Capture]
}

@Model
final class Capture {
    var session: CaptureSession?  // Plain var, no @Relationship
}
```

## Delete Rules Reference

### `.cascade` — Parent → Children
**Use when**: Parent owns children exclusively
**Effect**: Deleting parent deletes all children
**Examples in VisionCapture**:
- `Workflow` → `WorkflowStep` (deleting a workflow deletes its steps)
- `Workflow` → `WorkflowParameter` (deleting a workflow deletes its parameters)
- `CaptureSession` → `Capture` (deleting a session deletes its timeline frames)

### `.nullify` — Shared Relationships
**Use when**: Entities can exist independently of each other
**Effect**: Deleting one entity sets the relationship to nil on the other
**When this applies in VisionCapture**: use for any future many-to-many model such as a shared tag/label that can attach to multiple workflows without owning them.

### `.deny` — Prevent Deletion
**Use when**: Cannot delete parent if children exist
**Effect**: Deletion fails if children exist
**Rarely used in VisionCapture**

## VisionCapture Relationship Patterns

### Workflow as the Root Owner
```swift
@Model
final class Workflow {
    // Workflow owns its steps and parameters with cascade delete
    @Relationship(deleteRule: .cascade, inverse: \WorkflowStep.workflow)
    var steps: [WorkflowStep]

    @Relationship(deleteRule: .cascade, inverse: \WorkflowParameter.workflow)
    var parameters: [WorkflowParameter]
}
```

### Child Entities (Simple Reference)
```swift
@Model
final class WorkflowStep {
    // No @Relationship - just a plain reference
    var workflow: Workflow?
}

@Model
final class WorkflowParameter {
    // No @Relationship - just a plain reference
    var workflow: Workflow?
}
```

## Common Mistakes to Avoid

### ❌ WRONG: @Relationship on both sides
```swift
// Parent
@Relationship(deleteRule: .cascade, inverse: \WorkflowStep.workflow)
var steps: [WorkflowStep]

// Child - WRONG!
@Relationship(inverse: \Workflow.steps)  // ❌ Don't do this
var workflow: Workflow?
```

### ❌ WRONG: Missing inverse
```swift
@Relationship(deleteRule: .cascade)  // ❌ Missing inverse!
var steps: [WorkflowStep]
```

### ❌ WRONG: Incorrect inverse path
```swift
@Relationship(deleteRule: .cascade, inverse: \WorkflowStep.id)  // ❌ Wrong property!
var steps: [WorkflowStep]
```

### ❌ WRONG: Not using arrays for one-to-many
```swift
@Relationship(deleteRule: .cascade, inverse: \WorkflowStep.workflow)
var step: WorkflowStep?  // ❌ Should be [WorkflowStep]
```

## Checklist for New Relationships

Before creating a relationship, ask:

- [ ] **Who owns whom?** (Parent should have @Relationship)
- [ ] **What happens on delete?** (Choose .cascade or .nullify)
- [ ] **Is inverse correct?** (Points to the right property with correct keypath)
- [ ] **Is it one-to-many or many-to-many?** (Use correct delete rule)
- [ ] **Did I add comments?** (Explain the relationship and delete behavior)
- [ ] **Is child using plain var?** (No @Relationship on child for one-to-many)

## Documentation Template

Always use this comment template above relationships:

```swift
// Relationship: [Parent] has many [Children]
// Delete Rule: .[cascade/nullify/deny]
// Explanation: When a [Parent] is deleted, [what happens to children].
@Relationship(deleteRule: .cascade, inverse: \Child.parent)
var children: [Child]
```

## Production Warning

⚠️ **CRITICAL**: Relationships are IMMUTABLE in production!

Once a model with relationships is deployed:
- ❌ Cannot change delete rules
- ❌ Cannot change inverse relationships
- ❌ Cannot change cardinality (one-to-many ↔ many-to-many)
- ❌ Cannot rename relationship properties without migration

**Why**: The persisted store depends on stable relationship structure. Changes break existing user data (captured sessions, learned workflows, cached elements).

**If you MUST change**: Requires a lightweight or heavyweight migration and a user-data preservation plan.
Coordinate with the `swiftdata-specialist` role and get explicit user approval first.

## Reference Models (VisionCapture)

See these files in `/Users/dev-machine/Dev/VisionOS/VisionCapture/Sources/Core/Models/` for canonical patterns:
- `Workflow.swift` — parent with two cascade relationships (`steps`, `parameters`)
- `WorkflowStep.swift` — child with plain `var workflow: Workflow?`
- `WorkflowParameter.swift` — child with plain `var workflow: Workflow?`
- `Models.swift` — `CaptureSession` / `Capture` cascade pattern

The owning persistence managers live in `/Users/dev-machine/Dev/VisionOS/VisionCapture/Sources/Core/Persistence/` (`SwiftDataManager`, `WorkflowManager`, `CaptureManager`, etc.).

## Quick Reference Card

```
ONE-TO-MANY (Parent owns children):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parent:  @Relationship(deleteRule: .cascade, inverse: \Child.parent)
         var children: [Child]

Child:   var parent: Parent?  // Plain var, no @Relationship

MANY-TO-MANY (Shared, independent):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entity1: @Relationship(deleteRule: .nullify, inverse: \Entity2.entity1s)
         var entity2s: [Entity2]?

Entity2: var entity1s: [Entity1]?  // Plain var, no @Relationship
```

## Remember

1. **Parent has @Relationship, child has plain var** (for one-to-many)
2. **Always specify inverse** with correct keypath
3. **Use .cascade when parent owns children** exclusively
4. **Use .nullify for shared relationships**
5. **Document every relationship** with comments
6. **Relationships are immutable** in production - design carefully
