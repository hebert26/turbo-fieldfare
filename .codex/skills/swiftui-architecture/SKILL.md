---
name: swiftui-architecture
description: Expert guidance on SwiftUI architecture, ViewModel patterns, view composition, and state management. Auto-activates when working with SwiftUI views, ViewModels, navigation, or state. Enforces VisionCapture's Environment-based ViewModel pattern and stable view hierarchies for the macOS app at `/Users/dev-machine/Dev/VisionOS/VisionCapture/`.
---

# SwiftUI Architecture Skill

## Purpose
This skill provides comprehensive guidance on SwiftUI architecture patterns, ensuring correct ViewModel usage, stable view hierarchies, and proper state management in VisionCapture. It enforces VisionCapture's established patterns and prevents common mistakes. Target: macOS 26, Swift 6.2+, SwiftUI under `VisionCapture/Sources/Features/`, `UI/`, and `App/`.

## Auto-Trigger Conditions
This skill automatically activates when:
- Working with SwiftUI views, ViewModels, or view composition
- Using `@State`, `@Bindable`, `@Environment`, or `@Binding`
- Implementing navigation (NavigationStack, sheets, presentations)
- Creating or modifying view hierarchies
- Keywords: `SwiftUI`, `View`, `ViewModel`, `navigation`, `sheet`, `state management`

---

# TABLE OF CONTENTS

1. [ViewModel Pattern (The Three Rules)](#viewmodel-pattern)
2. [Stable View Hierarchies](#stable-hierarchies)
3. [State Management](#state-management)
4. [Navigation Patterns](#navigation)
5. [View Composition](#view-composition)
6. [Common Mistakes](#common-mistakes)
7. [Cross-References to Other Skills](#cross-references)
8. [Production Checklist](#checklist)

---

<a name="viewmodel-pattern"></a>
# VIEWMODEL PATTERN (THE THREE RULES)

## Critical: Environment-Based ViewModel Architecture

VisionCapture uses a specific ViewModel pattern that differs from traditional MVVM. **This is mandatory**.

### The Three Rules

| Rule | Description |
|------|-------------|
| **1. Initialize Once** | ViewModels are created at app startup in `VisionCapture/Sources/App/` (the app entry point) |
| **2. Access via @Environment** | Views access ViewModels through Environment, never as parameters |
| **3. @Bindable for Bindings** | Child views that need bindings use `@Bindable` |

---

## Rule 1: Initialize Once at App Startup

ViewModels are singleton-like objects created once when the app launches.

```swift
// VisionCapture/Sources/App/VisionCaptureApp.swift
@main
struct VisionCaptureApp: App {
    // Initialize ViewModels once at startup
    @State private var workflowsViewModel: WorkflowsViewModel
    @State private var captureViewModel: CaptureViewModel

    init() {
        let dataManager = SwiftDataManager.shared
        _workflowsViewModel = State(initialValue: WorkflowsViewModel(dataSource: dataManager))
        _captureViewModel = State(initialValue: CaptureViewModel(dataSource: dataManager))
    }

    var body: some Scene {
        WindowGroup {
            AppShellView()
                .environment(workflowsViewModel)  // Inject into environment
                .environment(captureViewModel)
        }
    }
}
```

**Why**:
- Single source of truth for app state
- ViewModels persist across navigation
- Prevents duplicate instances and state inconsistency

---

## Rule 2: Access via @Environment

Views retrieve ViewModels from the environment, never as constructor parameters.

```swift
// ✅ CORRECT: Access via @Environment
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        List(viewModel.filteredWorkflows) { workflow in
            WorkflowRow(workflow: workflow)
        }
        .task {
            await viewModel.updateFilteredWorkflows()
        }
    }
}

// ❌ WRONG: Passing as parameter
struct WorkflowsView: View {
    let viewModel: WorkflowsViewModel  // DON'T DO THIS

    var body: some View { ... }
}

// ❌ WRONG: Creating in view
struct WorkflowsView: View {
    @State private var viewModel = WorkflowsViewModel()  // DON'T DO THIS

    var body: some View { ... }
}
```

**Why**:
- Environment injection is SwiftUI's dependency injection pattern
- Allows any view in the hierarchy to access the ViewModel
- Prevents prop drilling through multiple view layers

---

## Rule 3: @Bindable for Bindings

When a child view needs to create bindings to ViewModel properties, use `@Bindable`.

```swift
// Parent view with @Environment
struct WorkflowsHubView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        // For child views that need bindings, pass with @Bindable
        @Bindable var bindableVM = viewModel

        NavigationStack {
            WorkflowsListView()
                .searchable(text: $bindableVM.searchText)
        }
    }
}

// Alternative: Child view declares @Bindable
struct EditWorkflowView: View {
    @Bindable var viewModel: WorkflowsViewModel

    var body: some View {
        Form {
            TextField("Title", text: $viewModel.workflowTitle)
            TextField("Intent", text: $viewModel.workflowIntent)
            TextEditor(text: $viewModel.workflowNotes)
        }
    }
}

// Parent passes it
struct ParentView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        EditWorkflowView(viewModel: viewModel)
    }
}
```

**Why**:
- `@Bindable` creates bindings from `@Observable` classes
- Required for TextField, Toggle, Picker, and other input controls
- Keeps ViewModel as single source of truth

---

## ViewModel Structure Pattern

```swift
import SwiftUI
import SwiftData
import OSLog

@Observable
class WorkflowsViewModel: @unchecked Sendable {
    // MARK: - Dependencies
    private let logger = Logger.workflow
    private let dataManager: SwiftDataManager

    // MARK: - UI State (non-persistent)
    var showingAddWorkflow = false
    var errorAlert: ErrorAlert?
    var successMessage: String?

    // MARK: - Search State
    var searchText: String = "" {
        didSet {
            isSearching = !searchText.isEmpty
        }
    }
    var isSearching = false

    // MARK: - Filter State
    var isFavoritesActive = false
    var selectedSortOption: SortOption = .dateAdded

    // MARK: - Computed Data (async updated)
    private(set) var filteredWorkflows: [Workflow] = []

    // MARK: - Refresh Counter (for computed property updates)
    private var refreshCounter = 0

    // MARK: - Initialization
    init(dataSource: SwiftDataManager) {
        self.dataManager = dataSource
    }

    // MARK: - Actions
    @MainActor
    func updateFilteredWorkflows() async {
        // Filter logic here
    }

    func toggleFavorite(_ workflow: Workflow) {
        // Action logic
        refreshCounter += 1  // Force UI update
    }
}
```

**Key Patterns**:
- `@Observable` macro (not `ObservableObject`)
- `@unchecked Sendable` for thread safety declaration
- `private(set)` for computed/filtered data
- `refreshCounter` pattern for forcing computed property updates
- `@MainActor` on async methods that update UI state

---

**Detailed documentation**: See `references/viewmodel-pattern.md`

---

<a name="stable-hierarchies"></a>
# STABLE VIEW HIERARCHIES

## Critical: Never Conditionally Swap Views

SwiftUI tracks view identity. Conditional view swapping breaks animations and state.

### The Rule: Use ZStack + Opacity

```swift
// ✅ CORRECT: Stable hierarchy with opacity
var body: some View {
    ZStack {
        // All states exist in hierarchy, controlled by opacity
        IdealStateView()
            .opacity(state == .ideal ? 1 : 0)

        EmptyStateView()
            .opacity(state == .empty ? 1 : 0)

        LoadingStateView()
            .opacity(state == .loading ? 1 : 0)

        ErrorStateView()
            .opacity(state == .error ? 1 : 0)
    }
    .animation(.snappy(duration: 0.3), value: state)
}
```

```swift
// ❌ WRONG: Conditional view swapping
var body: some View {
    if state == .loading {
        LoadingView()  // View identity changes!
    } else if state == .empty {
        EmptyView()    // Different view!
    } else {
        ContentView()  // Yet another view!
    }
}
```

**Why Conditional Swapping is Wrong**:
1. **View Identity**: SwiftUI creates new view instances on each condition change
2. **Broken Animations**: No smooth transition between states
3. **Lost State**: @State properties reset when view changes
4. **Performance**: SwiftUI can't diff views efficiently

---

## The Five States Pattern

Every view should design for these states:

| State | Description | Implementation |
|-------|-------------|----------------|
| **Ideal** | Everything working | Primary content view |
| **Empty** | No content yet | `ContentUnavailableView` with guidance |
| **Loading** | Processing | Skeleton screens or ProgressView |
| **Error** | Something failed | Error message + recovery action |
| **Success** | Action completed | Brief toast/feedback |

```swift
enum ViewState {
    case ideal
    case empty
    case loading
    case error(String)
}

struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
    @State private var state: ViewState = .loading

    var body: some View {
        ZStack {
            // Ideal: Show content
            workflowsList
                .opacity(state == .ideal ? 1 : 0)

            // Empty: No workflows
            emptyState
                .opacity(state == .empty ? 1 : 0)

            // Loading: Fetching data
            loadingState
                .opacity(state == .loading ? 1 : 0)

            // Error: Something went wrong
            errorState
                .opacity(state.isError ? 1 : 0)
        }
        .animation(.snappy(duration: 0.3), value: state)
        .task {
            await loadData()
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Workflows", systemImage: "rectangle.on.rectangle")
        } description: {
            Text("Workflows you capture will appear here.")
        } actions: {
            Button("Add Workflow") {
                viewModel.showingAddWorkflow = true
            }
            .buttonStyle(.borderedProminent)
        }
    }
}
```

---

**Detailed documentation**: See `references/view-patterns.md`

---

<a name="state-management"></a>
# STATE MANAGEMENT

## Property Wrapper Decision Tree

```
Do you need to store a value?
├─ YES: Is it a simple value (Bool, String, Int)?
│   ├─ YES: Is it owned by this view only?
│   │   └─ YES → @State
│   └─ NO: Is it an @Observable class?
│       ├─ YES: Is it shared across views?
│       │   ├─ YES → @Environment
│       │   └─ NO: Do you need bindings?
│       │       ├─ YES → @Bindable
│       │       └─ NO → let property
│       └─ NO → Consider making it @Observable
└─ NO: Are you receiving a value from parent?
    ├─ YES: Do you need to modify it?
    │   ├─ YES → @Binding
    │   └─ NO → let property
    └─ NO → You don't need a property wrapper
```

## Quick Reference Table

| Wrapper | Use Case | Example |
|---------|----------|---------|
| `@State` | View-local simple values | `@State private var isExpanded = false` |
| `@Binding` | Parent-child value passing | `@Binding var selectedItem: Item?` |
| `@Environment` | Shared ViewModel access | `@Environment(WorkflowsViewModel.self) var vm` |
| `@Bindable` | ViewModel bindings in child | `@Bindable var viewModel: MyViewModel` |
| `let` | Read-only @Observable | `let viewModel: MyViewModel` |

## Examples

### @State - View-Owned Simple Values
```swift
struct WorkflowRow: View {
    let workflow: Workflow
    @State private var isExpanded = false
    @State private var showingSheet = false

    var body: some View {
        VStack {
            Text(workflow.title)
            if isExpanded {
                Text(workflow.notes ?? "")
            }
        }
        .onTapGesture {
            withAnimation { isExpanded.toggle() }
        }
    }
}
```

### @Environment - Shared ViewModel
```swift
struct WorkflowsListView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        List(viewModel.filteredWorkflows) { workflow in
            WorkflowRow(workflow: workflow)
        }
    }
}
```

### @Bindable - ViewModel Bindings
```swift
struct SearchableWorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        @Bindable var bindableVM = viewModel

        NavigationStack {
            WorkflowsList()
                .searchable(text: $bindableVM.searchText)
                .onChange(of: viewModel.searchText) { _, newValue in
                    Task {
                        await viewModel.performSemanticSearch(newValue)
                    }
                }
        }
    }
}
```

### @Binding - Parent-Child Communication
```swift
struct FilterSheet: View {
    @Binding var selectedFilter: FilterOption
    @Binding var isPresented: Bool

    var body: some View {
        List {
            ForEach(FilterOption.allCases) { option in
                Button(option.title) {
                    selectedFilter = option
                    isPresented = false
                }
            }
        }
    }
}

// Parent
struct ParentView: View {
    @State private var filter: FilterOption = .all
    @State private var showingFilter = false

    var body: some View {
        Button("Filter") { showingFilter = true }
            .sheet(isPresented: $showingFilter) {
                FilterSheet(selectedFilter: $filter, isPresented: $showingFilter)
            }
    }
}
```

---

<a name="navigation"></a>
# NAVIGATION PATTERNS

## NavigationStack (Preferred)

```swift
struct LibraryView: View {
    @State private var navigationPath = NavigationPath()

    var body: some View {
        NavigationStack(path: $navigationPath) {
            List {
                NavigationLink("Workflows", value: Destination.workflows)
                NavigationLink("Captures", value: Destination.captures)
            }
            .navigationDestination(for: Destination.self) { destination in
                switch destination {
                case .workflows:
                    WorkflowsView()
                case .captures:
                    CapturesView()
                case .workflowDetail(let workflow):
                    WorkflowDetailView(workflow: workflow)
                }
            }
        }
    }

    enum Destination: Hashable {
        case workflows
        case captures
        case workflowDetail(Workflow)
    }
}
```

## Sheet Presentations

```swift
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        List { ... }
            .sheet(isPresented: $viewModel.showingAddWorkflow) {
                AddWorkflowView()
            }
            .sheet(item: $viewModel.selectedWorkflowForDetail) { workflow in
                WorkflowDetailView(workflow: workflow)
            }
    }
}
```

## Platform-Specific Presentations

```swift
struct WorkflowsView: View {
    @State private var showingSheet = false

    var body: some View {
        List { ... }
            .adaptiveSheet(isPresented: $showingSheet, preferredStyle: .sheet) {
                // Content
            }
    }
}

// Extension for Mac-friendly sheets
extension View {
    func adaptiveSheet<Content: View>(
        isPresented: Binding<Bool>,
        preferredStyle: UIModalPresentationStyle = .automatic,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        self.sheet(isPresented: isPresented) {
            content()
                .presentationDetents([.medium, .large])
        }
    }
}
```

---

<a name="view-composition"></a>
# VIEW COMPOSITION

## Extract Subviews for Clarity

```swift
// ✅ GOOD: Extracted subviews
struct WorkflowsView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                filterChips
                workflowsList
            }
            .navigationTitle("Workflows")
            .toolbar { toolbarContent }
        }
    }

    private var filterChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack { ... }
        }
    }

    private var workflowsList: some View {
        List { ... }
    }


    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button("Add") { ... }
        }
    }
}

// ❌ BAD: Everything inline
struct WorkflowsView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        // 50 lines of filter chips...
                    }
                }
                List {
                    // 100 lines of list content...
                }
            }
            .toolbar {
                // 30 lines of toolbar...
            }
        }
    }
}
```

## When to Create Separate View Files

| Scenario | Action |
|----------|--------|
| Reused in multiple places | Separate file |
| Complex logic (>50 lines) | Separate file |
| Has its own state management | Separate file |
| Simple, single-use | Private computed property |

---

<a name="common-mistakes"></a>
# COMMON MISTAKES

## 1. Traditional MVVM (FORBIDDEN)

```swift
// ❌ WRONG: Creating ViewModel in view
struct WorkflowsView: View {
    @State private var viewModel = WorkflowsViewModel()  // NO!
}

// ✅ CORRECT: Access via Environment
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
}
```

## 2. Conditional View Swapping (FORBIDDEN)

```swift
// ❌ WRONG: if/else views
var body: some View {
    if isLoading {
        ProgressView()
    } else {
        ContentView()
    }
}

// ✅ CORRECT: ZStack with opacity
var body: some View {
    ZStack {
        ProgressView().opacity(isLoading ? 1 : 0)
        ContentView().opacity(isLoading ? 0 : 1)
    }
}
```

## 3. Wrong Property Wrapper

```swift
// ❌ WRONG: @State with @Observable class
@State private var viewModel: WorkflowsViewModel  // NO!

// ✅ CORRECT: @Environment for shared ViewModels
@Environment(WorkflowsViewModel.self) private var viewModel
```

## 4. Passing ViewModel as Parameter

```swift
// ❌ WRONG: Prop drilling
struct ChildView: View {
    let viewModel: WorkflowsViewModel  // NO!
}

// ✅ CORRECT: Child accesses via Environment
struct ChildView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
}
```

## 5. Forgetting @Bindable

```swift
// ❌ WRONG: Can't create binding
struct EditView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        TextField("Title", text: $viewModel.title)  // ERROR!
    }
}

// ✅ CORRECT: Use @Bindable
struct EditView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel
        TextField("Title", text: $vm.title)  // Works!
    }
}
```

---

<a name="cross-references"></a>
# CROSS-REFERENCES TO OTHER SKILLS

## Concurrency Patterns
**See `swift-concurrency-expert`** for:
- `@MainActor` and ViewModel threading
- `async/await` patterns in SwiftUI
- `@Observable` macro deep dive
- `Task` management and `.task` modifier
- Actor isolation and `@unchecked Sendable`

## Data Model Relationships
**See `swiftdata-relationships`** for:
- `@Relationship` patterns (one-to-many, many-to-many)
- Delete rules (`.cascade`, `.nullify`)
- Parent-child model ownership
- CloudKit sync considerations

---

<a name="checklist"></a>
# PRODUCTION CHECKLIST

## ViewModel Pattern
- [ ] ViewModels initialized in `VisionCaptureApp.swift`
- [ ] Views access ViewModels via `@Environment`
- [ ] `@Bindable` used when bindings needed
- [ ] No ViewModel creation in views (`@State var vm = ...`)
- [ ] No ViewModel passing as parameters

## View Hierarchies
- [ ] ZStack + opacity for state changes (not if/else)
- [ ] All 5 states designed (ideal, empty, loading, error, success)
- [ ] Smooth animations with `.animation(.snappy, value:)`

## State Management
- [ ] Correct property wrapper for each use case
- [ ] `@State` only for view-local simple values
- [ ] `@Environment` for shared ViewModels
- [ ] `@Bindable` for ViewModel bindings
- [ ] `@Binding` for parent-child value passing

## Navigation
- [ ] NavigationStack for hierarchical navigation
- [ ] Sheets for modal content
- [ ] Platform-appropriate presentations

## View Composition
- [ ] Complex views extracted to subviews
- [ ] Reusable components in separate files
- [ ] Private computed properties for simple extractions

---

## Quick Reference Card

```swift
// ViewModel Pattern
@Environment(MyViewModel.self) private var viewModel  // Access
@Bindable var vm = viewModel  // For bindings

// Stable Hierarchies
ZStack {
    ViewA().opacity(condition ? 1 : 0)
    ViewB().opacity(condition ? 0 : 1)
}
.animation(.snappy(duration: 0.3), value: condition)

// State Management
@State private var localValue = false  // View-owned
@Binding var parentValue: Bool  // Parent-child
@Environment(VM.self) var vm  // Shared ViewModel
@Bindable var bindable = vm  // For bindings

// Navigation
NavigationStack(path: $path) { ... }
.sheet(isPresented: $show) { ... }
.sheet(item: $selected) { item in ... }
```

---

**Reference Files** (Codex loads as needed):
- `references/viewmodel-pattern.md` - Detailed ViewModel documentation
- `references/view-patterns.md` - View composition and hierarchy patterns
- `references/quick-reference.md` - Cheat sheet for quick lookup
