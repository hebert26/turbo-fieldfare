---
name: code-quality-enforcement
description: Enforces iOS/macOS 26 code quality standards. Validates 3-tier data architecture (SwiftDataManager → Domain Managers → ViewModels), centralized Logger extensions, proper error handling, and Apple patterns. Use after coding tasks or when user says "quality check", "validate", "enforce quality".
---

# Code Quality Enforcement

Validates and enforces code quality standards for iOS/macOS 26 projects.

## When to Use

- After major feature implementation
- Before git commits
- When user requests code review or quality validation

## Related Skills

**For SwiftUI-specific code, also apply checks from:**
- `swiftui-expert` skill (`.codex/skills/swiftui-expert/SKILL.md`)

When reviewing SwiftUI views, run the SwiftUI Expert validation commands in addition to the checks below. Key SwiftUI patterns to verify:
- `@Observable` over `ObservableObject`
- Modern API usage (`foregroundStyle()`, `clipShape()`, `NavigationStack`, etc.)
- `@State`/`@StateObject` marked `private`
- ForEach uses stable identity (not `.indices`)
- No `AnyView` in list rows
- `.task` modifier instead of `.onAppear { Task { } }`

## Report Output

**Reports MUST be saved to the project directory:**

```
<project-root>/enforcement-report/<YYYY-MM-DD-HHmm>-quality-report.md
```

**Example:**
```
/Users/dev-machine/Dev/VisionOS/enforcement-report/2026-01-23-1045-quality-report.md
```

Create the `enforcement-report/` directory if it doesn't exist. Use current datetime for the filename.

---

## CRITICAL: Mandatory Checks

**Every quality report MUST validate these in order:**

### 1. 🔴 Single SwiftDataManager Owns Container
- ONE class (SwiftDataManager) creates and owns the ModelContainer
- SwiftDataManager creates ALL domain managers with context injection
- **VIOLATION:** Multiple managers each creating their own ModelContainer

### 2. 🔴 Domain Managers Receive Context via Init
- Managers MUST have `init(context: ModelContext, logger: Logger)`
- Managers do NOT create their own ModelContainer
- **VIOLATION:** Manager with `static let shared` that creates its own container
- **VIOLATION:** `ModelContainer(...)` inside a domain manager's init

### 3. 🔴 SwiftData Access Encapsulated
- ALL `context.fetch/insert/delete` inside `*Manager.swift` files
- Managers expose business methods like `getAll()`, `add()`, `delete()`

### 4. 🔴 No Direct Context Access
- ViewModels/Views/MCP Tools call manager methods, NOT context directly
- **VIOLATION:** `@Environment(\.modelContext)` in a View
- **VIOLATION:** `context.fetch()` outside a manager

### 5. 🔴 Swift Concurrency (iOS 26 / Swift 6)
- ViewModels MUST be marked with `@MainActor`
- Use `@Observable` (not ObservableObject/`@Published`)
- Actors for shared mutable state (not classes with locks)
- SwiftUI views use `.task` modifier (not `.onAppear { Task { } }`)
- Long-running loops MUST check `Task.isCancelled`
- Use `MainActor.run` (not `DispatchQueue.main.async`)

**VIOLATIONS:**
```swift
// ❌ ViewModel missing @MainActor
@Observable
class WorkflowsViewModel {
    var items: [Workflow] = []
}

// ❌ Legacy ObservableObject
class ViewModel: ObservableObject {
    @Published var items: [Item] = []
}

// ❌ Class with manual locks for shared state
class Cache {
    private let lock = NSLock()
    private var storage: [String: Data] = [:]
}

// ❌ Task in onAppear (leaks, no auto-cancel)
.onAppear {
    Task {
        await loadData()
    }
}

// ❌ Loop without cancellation check
for item in items {
    await process(item)  // Continues even if cancelled
}

// ❌ Legacy DispatchQueue
DispatchQueue.main.async {
    self.updateUI()
}
```

**CORRECT:**
```swift
// ✅ ViewModel with @MainActor
@Observable
@MainActor
class WorkflowsViewModel {
    var items: [Workflow] = []
}

// ✅ Actor for shared mutable state
actor Cache {
    private var storage: [String: Data] = [:]

    func get(_ key: String) -> Data? { storage[key] }
    func set(_ key: String, value: Data) { storage[key] = value }
}

// ✅ .task modifier (auto-cancels on disappear)
.task {
    await loadData()
}

// ✅ Loop with cancellation check
for item in items {
    try Task.checkCancellation()
    await process(item)
}

// ✅ MainActor.run for UI updates
await MainActor.run {
    updateUI()
}

// ✅ defer for cleanup
func loadData() async {
    isLoading = true
    defer { isLoading = false }
    data = try await fetchData()
}
```

**Example of CORRECT architecture:**
```swift
// ONE SwiftDataManager owns container + creates managers
final class SwiftDataManager {
    let modelContainer: ModelContainer
    let workflowManager: WorkflowManager

    init() {
        self.modelContainer = try ModelContainer(...)
        self.workflowManager = WorkflowManager(context: modelContainer.mainContext, logger: Logger.workflow)
    }
}

// Manager receives context - doesn't create its own
final class WorkflowManager {
    private let context: ModelContext  // INJECTED
    init(context: ModelContext, logger: Logger) { self.context = context }
}
```

**Example of VIOLATION:**
```swift
// ❌ Manager creates its own container - WRONG
final class StorageWorkflowManager {
    static let shared = StorageWorkflowManager()
    private init() {
        modelContainer = try ModelContainer(...)  // VIOLATION!
    }
}
```

If any of these fail, the report must list them as **CRITICAL VIOLATIONS**.

---

## Core Architecture Rules

### 1. Three-Tier Data Architecture (MANDATORY)

```
┌─────────────────────────────────────────────────────────┐
│  APP CONTEXT (SwiftUI)                                   │
│  Views → @Environment(ViewModel.self)                   │
│  ViewModels → init(dataSource: .shared)                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  MCP/CLI CONTEXT (No SwiftUI)                            │
│  Tools → SwiftDataManager.shared.<domainManager>        │
│  (Direct singleton access OK - no Environment available) │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Domain Managers (init with context + logger)            │
│  Encapsulate ALL CRUD operations                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  SwiftDataManager (singleton, owns ModelContainer)       │
└─────────────────────────────────────────────────────────┘
```

**Violations to detect:**

```swift
// ❌ Direct context access in View
@Environment(\.modelContext) var context

// ❌ Direct fetch anywhere outside managers
let items = try context.fetch(descriptor)

// ❌ Direct insert outside manager
context.insert(item)
```

**Correct patterns - App Context (SwiftUI):**

```swift
// ✅ View uses ViewModel via Environment
@Environment(<Feature>ViewModel.self) var viewModel

// ✅ ViewModel uses domain manager
let items = try dataSource.<feature>Manager.getAll()
```

**Correct patterns - MCP/CLI Context (No SwiftUI):**

```swift
// ✅ MCP tool accesses via singleton (Environment not available)
func execute() async throws -> ToolResult {
    let items = try SwiftDataManager.shared.<feature>Manager.getAll()
    return .success(items)
}
```

**Correct patterns - Domain Manager:**

```swift
// ✅ Manager encapsulates CRUD
func add(_ item: <Model>) throws {
    context.insert(item)
    try saveContext()
}

// ✅ Private saveContext helper
private func saveContext() throws {
    if context.hasChanges {
        try context.save()
    }
}
```

### 2. Centralized Logger Extension

**Violation:**
```swift
// ❌ Inline Logger creation scattered everywhere
let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "app", category: "something")
```

**Correct pattern:**
```swift
// ✅ Centralized extension (one file, e.g., Logger+Extensions.swift)
extension Logger {
    static let subsystem = Bundle.main.bundleIdentifier ?? "YourAppName"

    // Add one static property per logical category in YOUR app
    // Categories should match your app's features/managers
    static let app = Logger(subsystem: subsystem, category: "app")
    static let <featureName> = Logger(subsystem: subsystem, category: "<FeatureName>")
    static let <managerName> = Logger(subsystem: subsystem, category: "<ManagerName>")
}

// ✅ Usage - use the category matching your feature
Logger.<featureName>.info("Operation completed")
```

**Note:** Replace `<featureName>` and `<managerName>` with VisionCapture's actual features. Examples:
- Core engine: `workflow`, `exploration`, `scout`, `recovery`
- Capture: `capture`, `ocr`, `annotation`
- Persistence: `storage`, `swiftData`, `cache`
- MCP/HTTP: `httpServer`, `mcp`, `session`
- Interaction: `interaction`, `wda`, `hotkey`

### 3. Singleton Pattern

**Acceptable for core infrastructure:**
```swift
// ✅ SwiftDataManager singleton
final class SwiftDataManager {
    @MainActor
    static let shared: SwiftDataManager = SwiftDataManager()

    // Domain managers for each feature
    let <feature>Manager: <Feature>Manager

    @MainActor
    init() {
        // Create managers with context + logger injection
        self.<feature>Manager = <Feature>Manager(
            context: modelContext,
            logger: Logger.<feature>
        )
    }
}
```

**Not acceptable for:**
- ViewModels (use Environment injection)
- Domain managers (use init injection)

### 4. Error Handling

**Critical failures → log + fatalError:**
```swift
// ✅ Correct for unrecoverable infrastructure failures
guard let appSupportURL = FileManager.default.urls(
    for: .applicationSupportDirectory,
    in: .userDomainMask
).first else {
    logger.error("Failed to access application support directory")
    fatalError("Unable to access application support directory")
}
```

**Non-critical → try? is acceptable:**
```swift
// ✅ Acceptable - backup exclusion is not critical
try? url.setResourceValues(resourceValues)
```

**Recoverable failures → propagate with throws:**
```swift
// ✅ Domain manager propagates errors
func getAll() throws -> [<Model>] {
    guard let requiredDependency = try context.fetchRequired() else {
        logger.error("Required dependency not found")
        throw Errors.dependencyNotFound
    }
    let descriptor = FetchDescriptor<<Model>>()
    return try context.fetch(descriptor)
}
```

### 5. Debug Logging

```swift
// ✅ Verbose logging only in DEBUG
#if DEBUG
logger.debug("Fetched \(items.count) items")
#endif

// ✅ Important logs always (info, warning, error, critical)
logger.info("Manager initialized successfully")
logger.error("Failed to save context: \(error.localizedDescription, privacy: .public)")
```

### 6. Privacy Levels

```swift
// ✅ Always specify privacy on interpolations
logger.info("Name: \(name, privacy: .public)")        // Safe to expose
logger.debug("ID: \(id, privacy: .private)")          // Redacted in release
logger.critical("Error: \(error.localizedDescription, privacy: .auto)")
```

## Validation Commands

### 🔴 CRITICAL: Architecture Pattern (Run First!)

```bash
# 1. Find managers that create their own ModelContainer (VIOLATION!)
grep -rn "ModelContainer(" --include="*Manager.swift" .
# Expected: ONLY SwiftDataManager.swift should have ModelContainer(

# 2. Check if managers receive context via init (CORRECT pattern)
grep -rn "init(context: ModelContext" --include="*Manager.swift" .
# Expected: All domain managers should have this pattern

# 3. Find managers with static let shared that own a container (VIOLATION!)
grep -rn "static let shared" --include="*Manager.swift" .
# Then check if those files also have ModelContainer( - that's a violation

# 4. Find direct modelContext usage in Views (VIOLATION)
grep -rn "@Environment(\\.modelContext)" --include="*.swift" .
# Expected: NO matches

# 5. Find context operations OUTSIDE manager files (VIOLATION)
grep -rn "context\.fetch\|context\.insert\|context\.delete" --include="*.swift" . | grep -v "Manager\.swift"
# Expected: NO matches
```

**Expected architecture:**
- ONE `SwiftDataManager` with `ModelContainer(`
- Domain managers with `init(context: ModelContext, logger: Logger)`
- NO `ModelContainer(` in domain managers

### 🔴 CRITICAL: No print() Statements

```bash
# Find print statements (MUST use Logger instead)
grep -rn "print(" --include="*.swift" . | grep -v "#if DEBUG" | grep -v "Test"
# Expected: NO matches outside of #if DEBUG blocks
```

**VIOLATION:** Any `print()` statement in production code
**CORRECT:** Use `Logger.<category>.info/debug/error()` instead

### 🟡 Other Checks

```bash
# Find inline Logger creation (should use extension)
grep -rn "Logger(subsystem:" --include="*.swift" . | grep -v "extension Logger"

# Find empty catch blocks
grep -rn "catch { }" --include="*.swift" .

# Check Logger extension exists
grep -rn "extension Logger" --include="*.swift" .
```

### 🔴 Swift Concurrency Checks

```bash
# Find ViewModels missing @MainActor (CRITICAL)
grep -rn "class.*ViewModel" --include="*.swift" . | grep -v "@MainActor"

# Find legacy ObservableObject usage (should use @Observable)
grep -rn "ObservableObject\|@Published\|@StateObject\|@ObservedObject" --include="*.swift" .

# Find DispatchQueue.main usage (should use MainActor.run)
grep -rn "DispatchQueue\.main" --include="*.swift" .

# Find .onAppear with Task (should use .task modifier)
grep -rn "onAppear" --include="*.swift" . | grep -i "task"

# Find for-await loops (verify they check cancellation)
grep -rn "for.*await\|for try await" --include="*.swift" .

# Find classes with NSLock/DispatchQueue for state (should use actor)
grep -rn "NSLock\|DispatchSemaphore" --include="*.swift" . | grep -v "unchecked Sendable"
```

### 🔴 CRITICAL: Crash & Security Risks

```bash
# 1. Find try! (crashes on failure)
grep -rn "try!" --include="*.swift" .
# Expected: NO matches

# 2. Find as! (crashes on type mismatch)
grep -rn "as!" --include="*.swift" .
# Expected: NO matches

# 3. Find force unwrap ! (crashes on nil) - exclude IBOutlet, try!, as!
grep -rn "\w!" --include="*.swift" . | grep -v "IBOutlet\|try!\|as!\|!=\|!="
# Expected: Minimal matches, review each

# 4. Find [unowned self] (crashes if deallocated)
grep -rn "\[unowned self\]" --include="*.swift" .
# Expected: NO matches (use [weak self] instead)

# 5. Find hardcoded secrets (security breach)
grep -rn "api_key\|apiKey\|secret\|password\|token" --include="*.swift" . | grep -i "=.*\""
# Expected: NO matches with hardcoded string values

# 6. Find Thread.sleep / sleep() (blocks thread)
grep -rn "Thread\.sleep\|sleep(" --include="*.swift" .
# Expected: NO matches (use Task.sleep or async patterns)
```

### 🟠 WARNING: Memory Leak Risks

```bash
# 7. Find closures with self but missing [weak self]
grep -rn "{ self\." --include="*.swift" . | grep -v "\[weak self\]\|\[unowned self\]"
# Expected: Review each - may need [weak self]

# 8. Check Timer usage (must have invalidate() in deinit)
grep -rn "Timer\." --include="*.swift" .
# Then verify: each Timer has corresponding invalidate() in deinit

# 9. Check NotificationCenter.addObserver (must have removeObserver)
grep -rn "NotificationCenter.*addObserver\|\.addObserver(" --include="*.swift" .
# Then verify: each addObserver has corresponding removeObserver
```

### 🟡 CODE QUALITY

```bash
# 10. Find TODO/FIXME comments (tech debt)
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.swift" .
# Expected: Track and address these

# 11. Find commented-out code (dead code)
grep -rn "^[[:space:]]*//.*func\|^[[:space:]]*//.*let\|^[[:space:]]*//.*var" --include="*.swift" .
# Expected: Remove commented-out code

# 12. Find files > 500 lines (maintainability warning)
find . -name "*.swift" -exec wc -l {} \; | awk '$1 > 500 {print}'
# Expected: Consider splitting large files or justify why they stay together

# 12b. Find files > 1000 lines (hard ceiling)
find . -name "*.swift" -exec wc -l {} \; | awk '$1 > 1000 {print}'
# Expected: NO matches unless the user explicitly accepts an exception

# 13. Find magic numbers (readability)
grep -rn "[^0-9][0-9]\{2,\}[^0-9]" --include="*.swift" . | grep -v "Test\|\.0\|import\|@available"
# Expected: Replace with named constants
```

### 🟣 LEGACY PATTERNS

```bash
# 14. Find Any / AnyObject usage (type safety)
grep -rn ": Any\|: AnyObject\|as Any\|as AnyObject" --include="*.swift" .
# Expected: Use specific types or protocols

# 15. Find CFRunLoop / performSelector (legacy/unsafe)
grep -rn "CFRunLoop\|performSelector" --include="*.swift" .
# Expected: NO matches (use async/await)
```

## Checklist

### 🔴 CRITICAL: Architecture Pattern
- [ ] **ONE** SwiftDataManager owns ModelContainer
- [ ] SwiftDataManager creates ALL domain managers with context injection
- [ ] Domain managers have `init(context: ModelContext, logger: Logger)`
- [ ] Domain managers do NOT have `ModelContainer(` in their code
- [ ] Domain managers do NOT have `static let shared` with their own container
- [ ] ALL `context.fetch/insert/delete` calls are inside `*Manager.swift` files
- [ ] ViewModels/Views/Tools call manager methods, NOT context directly
- [ ] No `@Environment(\.modelContext)` in Views

### 🔴 CRITICAL: Swift Concurrency
- [ ] ALL ViewModels marked with `@MainActor`
- [ ] `@Observable` used (not ObservableObject/`@Published`)
- [ ] Actors used for shared mutable state
- [ ] SwiftUI `.task` modifier used (not `.onAppear { Task { } }`)
- [ ] Long loops check `Task.isCancelled`
- [ ] No `DispatchQueue.main.async` in codebase

### 🔴 CRITICAL: Crash & Security Risks
- [ ] No try! (crashes on failure)
- [ ] No as! (crashes on type mismatch)
- [ ] No force unwrap on optionals (crashes on nil)
- [ ] No [unowned self] (use [weak self] instead)
- [ ] No hardcoded secrets (api_key, password, token in strings)
- [ ] No Thread.sleep or sleep() (use Task.sleep)

### 🟠 WARNING: Memory Leak Risks
- [ ] Closures capturing self have `[weak self]`
- [ ] Timer instances have `invalidate()` in deinit
- [ ] NotificationCenter observers have `removeObserver`

### 🟡 CODE QUALITY
- [ ] TODO/FIXME comments tracked and addressed
- [ ] No commented-out code blocks
- [ ] Files over 500 lines reviewed for splitting or explicitly justified
- [ ] Files under 1,000 lines unless the user explicitly accepts an exception
- [ ] Magic numbers replaced with named constants

### 🟣 LEGACY PATTERNS
- [ ] No `Any` / `AnyObject` (use specific types)
- [ ] No `CFRunLoop` / `performSelector` (use async/await)

### Data Architecture
- [ ] SwiftDataManager owns ModelContainer (singleton OK)
- [ ] Domain managers receive context + logger via init
- [ ] **App context:** ViewModels via @Environment, init with dataSource: .shared
- [ ] **MCP/CLI context:** Direct SwiftDataManager.shared.manager access OK
- [ ] Private saveContext() helper in managers

**Logging**
- [ ] 🔴 **No `print()` statements** (use Logger instead)
- [ ] Centralized `extension Logger` exists with static categories
- [ ] All logging uses `Logger.category` (not inline creation)
- [ ] Privacy levels on all interpolations
- [ ] Verbose logs wrapped in `#if DEBUG`

**Error Handling**
- [ ] Critical failures: log + fatalError (DB init, directory access)
- [ ] Recoverable failures: throws propagation
- [ ] Non-critical operations: try? acceptable
- [ ] No empty catch blocks
- [ ] Typed error enums with LocalizedError

**Apple Patterns**
- [ ] @Observable (not ObservableObject) for iOS 17+
- [ ] Versioned SwiftData schemas

**Swift Concurrency**
- [ ] ViewModels marked with `@MainActor`
- [ ] `@MainActor` on SwiftDataManager.shared property
- [ ] Actors protect shared mutable state (not classes with locks)
- [ ] SwiftUI views use `.task` modifier (not `.onAppear { Task { } }`)
- [ ] Long-running loops check `Task.isCancelled` or `try Task.checkCancellation()`
- [ ] No `DispatchQueue.main.async` (use `await MainActor.run`)
- [ ] Sendable types for cross-actor sharing
- [ ] `defer { }` pattern for cleanup (e.g., `isLoading = false`)
- [ ] Async functions use `async throws` (not completion handlers)

## Domain Manager Template

Replace `<Feature>` and `<Model>` with your actual feature/model names.

```swift
final class <Feature>Manager {
    private let context: ModelContext
    private let logger: Logger

    init(context: ModelContext, logger: Logger = Logger.<feature>) {
        self.context = context
        self.logger = logger
    }

    func getAll() throws -> [<Model>] {
        let descriptor = FetchDescriptor<<Model>>()
        let items = try context.fetch(descriptor)
        #if DEBUG
        logger.debug("Fetched \(items.count) items")
        #endif
        return items
    }

    func add(_ item: <Model>) throws {
        context.insert(item)
        try saveContext()
    }

    func delete(_ item: <Model>) throws {
        context.delete(item)
        try saveContext()
    }

    private func saveContext() throws {
        if context.hasChanges {
            try context.save()
        }
    }
}
```

**Examples (all from VisionCapture `Sources/Core/Persistence/`):**
- `WorkflowManager` + `Workflow`
- `CaptureManager` + `Capture` / `CaptureSession`
- `IdempotentRuleManager` + `IdempotentRule`
- `LearningCacheManager` + `CachedElement`
