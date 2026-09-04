# View Patterns Reference

## Stable View Hierarchies, Composition, and Navigation

This document provides comprehensive guidance on SwiftUI view patterns that ensure smooth animations, proper state management, and clean architecture.

**Official Documentation**:
- [SwiftUI Documentation](https://developer.apple.com/documentation/swiftui)
- [Getting Started with SwiftUI](https://www.swift.org/getting-started/swiftui/)

---

## Stable View Hierarchies

### The Problem: View Identity

SwiftUI uses **structural identity** to track views. When you use `if/else`, SwiftUI sees different view types:

```swift
// ❌ WRONG: Different view types
var body: some View {
    if isLoading {
        ProgressView()  // Type: ProgressView
    } else {
        ContentView()   // Type: ContentView
    }
}
```

**What happens**:
1. SwiftUI destroys `ProgressView`
2. Creates entirely new `ContentView`
3. No animation between states
4. @State in child views resets
5. Poor performance from view recreation

### The Solution: ZStack + Opacity

```swift
// ✅ CORRECT: Same view types, controlled by opacity
var body: some View {
    ZStack {
        ProgressView()
            .opacity(isLoading ? 1 : 0)

        ContentView()
            .opacity(isLoading ? 0 : 1)
    }
    .animation(.snappy(duration: 0.3), value: isLoading)
}
```

**What happens**:
1. Both views exist in hierarchy
2. Only opacity changes
3. Smooth cross-fade animation
4. @State preserved in both views
5. Efficient - no view recreation

---

## The Five States Pattern

Every view should handle these states:

### 1. Ideal State
Content loaded and displayed normally.

```swift
private var idealState: some View {
    List(viewModel.items) { item in
        ItemRow(item: item)
    }
    .listStyle(.plain)
    .refreshable {
        await viewModel.reload()
    }
}
```

### 2. Empty State
No content to display.

```swift
private var emptyState: some View {
    ContentUnavailableView {
        Label("No Items", systemImage: "tray")
    } description: {
        Text("Items you create will appear here.")
    } actions: {
        Button("Create Item") {
            viewModel.showingAddSheet = true
        }
        .buttonStyle(.borderedProminent)
    }
}
```

### 3. Loading State
Data is being fetched.

```swift
private var loadingState: some View {
    VStack(spacing: 16) {
        ForEach(0..<5, id: \.self) { _ in
            SkeletonRow()
        }
    }
    .padding()
}

struct SkeletonRow: View {
    @State private var isAnimating = false

    var body: some View {
        HStack(spacing: 16) {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.3))
                .frame(width: 44, height: 44)

            VStack(alignment: .leading, spacing: 8) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: 16)
                    .frame(maxWidth: 200)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: 12)
                    .frame(maxWidth: 120)
            }
        }
        .overlay(shimmerOverlay)
        .onAppear { isAnimating = true }
    }

    private var shimmerOverlay: some View {
        LinearGradient(
            colors: [.clear, .white.opacity(0.3), .clear],
            startPoint: .leading,
            endPoint: .trailing
        )
        .frame(width: 80)
        .offset(x: isAnimating ? 300 : -300)
        .animation(
            .linear(duration: 1.5).repeatForever(autoreverses: false),
            value: isAnimating
        )
    }
}
```

### 4. Error State
Something went wrong.

```swift
private var errorState: some View {
    ContentUnavailableView {
        Label("Connection Error", systemImage: "wifi.slash")
    } description: {
        Text(viewModel.errorMessage ?? "Unable to load data.")
    } actions: {
        Button("Try Again") {
            Task { await viewModel.reload() }
        }
        .buttonStyle(.borderedProminent)
    }
}
```

### 5. Success State
Action completed successfully.

```swift
@ViewBuilder
private var successOverlay: some View {
    if viewModel.showingSuccessToast {
        VStack {
            Spacer()

            HStack(spacing: 12) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .symbolEffect(.bounce, value: viewModel.showingSuccessToast)

                Text(viewModel.successMessage ?? "Success")
                    .font(.subheadline)
            }
            .padding(16)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .shadow(color: .black.opacity(0.1), radius: 8, y: 4)
            .padding(.bottom, 32)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }
}
```

### Complete Implementation

```swift
struct ItemsView: View {
    @Environment(ItemsViewModel.self) private var viewModel

    var body: some View {
        NavigationStack {
            ZStack {
                // All states in hierarchy
                idealState
                    .opacity(viewModel.state == .ideal ? 1 : 0)

                emptyState
                    .opacity(viewModel.state == .empty ? 1 : 0)

                loadingState
                    .opacity(viewModel.state == .loading ? 1 : 0)

                errorState
                    .opacity(viewModel.state == .error ? 1 : 0)

                // Success overlay (on top)
                successOverlay
            }
            .animation(.snappy(duration: 0.3), value: viewModel.state)
            .navigationTitle("Items")
            .task {
                await viewModel.loadItems()
            }
        }
    }

    // ... state views defined above
}
```

---

## View Composition Patterns

### 1. Extract Computed Properties

For simple, view-specific extractions:

```swift
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                filterChips      // Extracted
                workflowsList    // Extracted
            }
            .navigationTitle("Workflows")
            .toolbar { toolbarContent }  // Extracted
        }
    }

    // MARK: - Subviews

    private var filterChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                FilterChip(title: "All", isSelected: !viewModel.hasActiveFilters)
                FilterChip(title: "Favorites", isSelected: viewModel.isFavoritesActive)
                FilterChip(title: "Pinned", isSelected: viewModel.isPinnedActive)
            }
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }

    private var workflowsList: some View {
        List(viewModel.filteredWorkflows) { workflow in
            WorkflowRow(workflow: workflow)
        }
        .listStyle(.plain)
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button {
                viewModel.showingAddWorkflow = true
            } label: {
                Image(systemName: "plus")
            }
        }

        ToolbarItem(placement: .topBarLeading) {
            Menu {
                sortMenuContent
            } label: {
                Image(systemName: "arrow.up.arrow.down")
            }
        }
    }

    @ViewBuilder
    private var sortMenuContent: some View {
        ForEach(SortOption.allCases) { option in
            Button {
                viewModel.selectedSortOption = option
            } label: {
                HStack {
                    Text(option.title)
                    if viewModel.selectedSortOption == option {
                        Image(systemName: "checkmark")
                    }
                }
            }
        }
    }
}
```

### 2. Separate View Files

For reusable or complex components:

```
VisionCapture/Sources/Features/Workflows/Views/
├── WorkflowsHubView.swift      # Main container
├── WorkflowsListView.swift     # List component
├── WorkflowRow.swift           # Row component (reused)
├── WorkflowDetailView.swift    # Detail sheet
├── AddWorkflowView.swift       # Add sheet
└── Components/
    ├── FilterChip.swift        # Reusable chip
    └── SkeletonRow.swift       # Loading skeleton
```

### 3. When to Extract

| Scenario | Action |
|----------|--------|
| Used in multiple views | Separate file in `Components/` |
| Complex (>50 lines) | Separate file |
| Has own state/logic | Separate file |
| Simple, single-use | Private computed property |
| Toolbar content | `@ToolbarContentBuilder` property |
| Menu content | `@ViewBuilder` property |

---

## Navigation Patterns

### NavigationStack (Primary Navigation)

```swift
struct LibraryView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            List {
                Section("Library") {
                    NavigationLink(value: Destination.workflows) {
                        Label("Workflows", systemImage: "rectangle.on.rectangle")
                    }
                    NavigationLink(value: Destination.captures) {
                        Label("Captures", systemImage: "camera")
                    }
                }
            }
            .navigationTitle("Library")
            .navigationDestination(for: Destination.self) { destination in
                destinationView(for: destination)
            }
        }
    }

    @ViewBuilder
    private func destinationView(for destination: Destination) -> some View {
        switch destination {
        case .workflows:
            WorkflowsHubView()
        case .captures:
            CapturesHubView()
        case .workflowDetail(let workflow):
            WorkflowDetailView(workflow: workflow)
        case .captureDetail(let capture):
            CaptureDetailView(capture: capture)
        }
    }
}

enum Destination: Hashable {
    case workflows
    case captures
    case workflowDetail(Workflow)
    case captureDetail(Capture)
}
```

### Programmatic Navigation

```swift
struct WorkflowsView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            List(workflows) { workflow in
                WorkflowRow(workflow: workflow)
                    .onTapGesture {
                        // Programmatic navigation
                        path.append(Destination.workflowDetail(workflow))
                    }
            }
            .navigationDestination(for: Destination.self) { ... }
        }
    }

    // Pop to root
    func popToRoot() {
        path.removeLast(path.count)
    }

    // Pop one level
    func popOne() {
        if !path.isEmpty {
            path.removeLast()
        }
    }
}
```

### Sheet Presentations

```swift
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        List { ... }
            // Boolean-based sheet
            .sheet(isPresented: $viewModel.showingAddWorkflow) {
                AddWorkflowView()
            }

            // Item-based sheet
            .sheet(item: $viewModel.selectedWorkflowForDetail) { workflow in
                WorkflowDetailView(workflow: workflow)
            }

            // Full-screen cover
            .fullScreenCover(isPresented: $viewModel.showingFullScreen) {
                FullScreenView()
            }
    }
}
```

### Confirmation Dialogs

```swift
struct WorkflowRow: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
    let workflow: Workflow
    @State private var showingDeleteConfirmation = false

    var body: some View {
        HStack { ... }
            .swipeActions(edge: .trailing) {
                Button(role: .destructive) {
                    showingDeleteConfirmation = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
            .confirmationDialog(
                "Delete Workflow",
                isPresented: $showingDeleteConfirmation,
                titleVisibility: .visible
            ) {
                Button("Delete", role: .destructive) {
                    viewModel.deleteWorkflow(workflow)
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Are you sure you want to delete '\(workflow.title)'?")
            }
    }
}
```

### Alert Presentations

```swift
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        List { ... }
            .alert(
                item: $viewModel.errorAlert,
                title: { Text($0.title) },
                actions: { _ in
                    Button("OK") {}
                },
                message: { alert in
                    Text(alert.message)
                }
            )
    }
}

// Or simpler:
.alert(
    "Error",
    isPresented: $showingError,
    presenting: errorMessage
) { _ in
    Button("OK") {}
} message: { message in
    Text(message)
}
```

---

## Platform-Specific Patterns

### Adaptive Layouts

```swift
struct WorkflowsView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        if horizontalSizeClass == .regular {
            // iPad/Mac: Two-column layout
            NavigationSplitView {
                sidebarContent
            } detail: {
                detailContent
            }
        } else {
            // iPhone: Stack layout
            NavigationStack {
                listContent
            }
        }
    }
}
```

### Mac-Specific Adjustments

```swift
struct WorkflowsView: View {
    var body: some View {
        List { ... }
            #if os(macOS)
            .listStyle(.sidebar)
            #else
            .listStyle(.plain)
            #endif
    }
}

// Or using UIDevice
extension UIDevice {
    var isRunningOnMac: Bool {
        UIDevice.current.userInterfaceIdiom == .pad &&
        ProcessInfo.processInfo.isiOSAppOnMac
    }
}

struct AdaptiveView: View {
    var body: some View {
        Group {
            if UIDevice.current.isRunningOnMac {
                macContent
            } else {
                iosContent
            }
        }
    }
}
```

### Adaptive Sheet Presentation

```swift
extension View {
    func adaptiveSheet<Content: View>(
        isPresented: Binding<Bool>,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        self.sheet(isPresented: isPresented) {
            content()
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
    }
}

// Usage
.adaptiveSheet(isPresented: $showingSheet) {
    SheetContent()
}
```

---

## Animation Patterns

### State-Based Animations

```swift
struct AnimatedView: View {
    @State private var isExpanded = false

    var body: some View {
        VStack {
            Button("Toggle") {
                withAnimation(.snappy(duration: 0.3)) {
                    isExpanded.toggle()
                }
            }

            if isExpanded {
                ExpandedContent()
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
            }
        }
    }
}
```

### Value-Based Animations

```swift
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        ZStack {
            contentView
            loadingView.opacity(viewModel.isLoading ? 1 : 0)
        }
        .animation(.snappy(duration: 0.3), value: viewModel.isLoading)
        .animation(.snappy(duration: 0.3), value: viewModel.state)
    }
}
```

### Symbol Effects (iOS 17+)

```swift
Image(systemName: "heart.fill")
    .symbolEffect(.bounce, value: isFavorite)

Image(systemName: "checkmark.circle.fill")
    .symbolEffect(.pulse, options: .repeating)

Image(systemName: "arrow.clockwise")
    .symbolEffect(.rotate, value: isRefreshing)
```

---

## Common Mistakes

### 1. Conditional View Swapping

```swift
// ❌ WRONG
if condition {
    ViewA()
} else {
    ViewB()
}

// ✅ CORRECT
ZStack {
    ViewA().opacity(condition ? 1 : 0)
    ViewB().opacity(condition ? 0 : 1)
}
```

### 2. Heavy Work in body

```swift
// ❌ WRONG
var body: some View {
    let filtered = items.filter { ... }  // Called every render!
    List(filtered) { ... }
}

// ✅ CORRECT
// Use ViewModel's async filtering
var body: some View {
    List(viewModel.filteredItems) { ... }
}
```

### 3. Inline Complex Views

```swift
// ❌ WRONG: 200 lines inline
var body: some View {
    VStack {
        // 200 lines of nested views...
    }
}

// ✅ CORRECT: Extracted
var body: some View {
    VStack {
        headerSection
        contentSection
        footerSection
    }
}

private var headerSection: some View { ... }
private var contentSection: some View { ... }
private var footerSection: some View { ... }
```

### 4. Missing Animation Values

```swift
// ❌ WRONG: Animates everything
.animation(.snappy)

// ✅ CORRECT: Specific value
.animation(.snappy(duration: 0.3), value: isExpanded)
```

---

## Summary

| Pattern | Use Case |
|---------|----------|
| **ZStack + Opacity** | State transitions (loading/error/content) |
| **Five States** | Complete UX coverage |
| **Computed Properties** | Simple view extractions |
| **Separate Files** | Reusable/complex components |
| **NavigationStack** | Hierarchical navigation |
| **Sheets** | Modal content |
| **Value Animations** | Smooth, targeted transitions |
