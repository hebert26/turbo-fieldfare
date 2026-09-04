# UML Diagram Type Reference

Use this file to choose a diagram by the question it answers. UML has 14 commonly presented types in two broad groups: structural diagrams describe what exists; behavioral diagrams describe actions, interactions, and change over time. The categories and practical uses below are a concise working summary of Visual Paradigm's guide, not a replacement for a formal UML specification.

## Behavioral diagrams

| Type | Model | Use it when | Essential elements |
| --- | --- | --- | --- |
| Activity | Control and object flow through work | Mapping a process, workflow, approval chain, branch, loop, or parallel work | Actions, initial/final nodes, decisions, merges, forks/joins, optional swimlanes |
| Use Case | External goals and system responsibilities | Defining scope and requirements before implementation detail | Actors, system boundary, verb-based use cases, associations, include/extend where justified |
| Sequence | Chronological messages between participants | Explaining an API call chain, user journey, service interaction, or failure response | Participants/lifelines, ordered messages, activations, returns, alternatives, loops |
| State Machine | Discrete states and event-triggered transitions | Modeling an order, account, workflow, session, device, or other lifecycle | States, events, guards, actions, initial/final or terminal states |
| Communication | Object collaboration with numbered messages | Showing who is connected to whom when topology matters more than exact time order | Participants, links, numbered messages, direction, collaboration context |
| Interaction Overview | High-level control flow over referenced interactions | Orchestrating a large process that needs activity-style routing plus drill-down sequence views | Initial/final nodes, decisions, forks/joins, interaction references, notes |
| Timing | State/value change along an explicit time axis | Specifying deadlines, durations, real-time behavior, protocol timing, or SLA constraints | Lifelines/tracks, time axis, state/value changes, duration and time constraints |

## Structural diagrams

| Type | Model | Use it when | Essential elements |
| --- | --- | --- | --- |
| Class | Types, data, behavior, and static relationships | Describing a domain model, object-oriented design, or stable interface contract | Classes/interfaces, attributes, operations, associations, inheritance, composition, dependencies, multiplicity |
| Object | Concrete instances at one point in time | Validating a class model with a scenario or showing actual values and links | Instance names, classifier types, attribute values, links |
| Component | Replaceable implementation units and contracts | Showing modules, services, libraries, APIs, and dependency boundaries | Components, provided/required interfaces, ports where useful, dependencies |
| Composite Structure | Internal parts and collaboration inside one classifier | Explaining a component's internal wiring, ports, extension points, or complex framework mechanism | Structured classifier, parts, ports, connectors, roles |
| Deployment | Software mapped to runtime or physical infrastructure | Planning hosts, devices, containers, artifacts, network paths, scaling, or failover | Nodes, execution environments, artifacts, communication paths, protocols |
| Package | Logical grouping and dependency direction | Organizing a large model or codebase and explaining architectural layers | Packages/namespaces, contained elements, dependency arrows |
| Profile | Extensions to UML for a specialized domain | Defining stereotypes, tagged values, or constraints for a reusable modeling profile | Profile package, stereotypes, metaclasses, tagged values, constraints |

## Selection heuristics

Use this short mapping when the task is phrased as a goal:

| Task wording | First choice | Consider instead |
| --- | --- | --- |
| “Who uses this and what can they do?” | Use Case | Activity for the detailed flow |
| “How does this process work?” | Activity | Sequence for service messages; State Machine for lifecycle |
| “What calls what, and in what order?” | Sequence | Communication if topology is the main point |
| “What states can this entity be in?” | State Machine | Timing when durations and deadlines are central |
| “What are the entities and relationships?” | Class | Object for one concrete scenario |
| “What modules depend on each other?” | Component | Package for logical organization; Deployment for where they run |
| “Where does each service run?” | Deployment | Component for service boundaries without infrastructure |
| “Show the whole orchestration with detailed interaction steps” | Interaction Overview | Activity for a standalone workflow |
| “Show the internal pieces and ports of this component” | Composite Structure | Component for external boundaries |
| “Create a custom notation for our domain” | Profile | Use stereotypes in another UML diagram only if the profile is already defined |

## Notation and format guidance

- Mermaid is useful for activity-like flows, sequence diagrams, state diagrams, class diagrams, architecture/component-like views, and simple entity relationships. Treat its syntax as a practical representation, not proof that every UML semantic is preserved.
- PlantUML is a better default for precise UML notation, especially use cases, communication diagrams, composite structures, deployment diagrams, profiles, guards, stereotypes, and detailed class relationships.
- If the user requests a specific renderer, obey that request and check its supported syntax. Do not silently substitute Mermaid for PlantUML or vice versa.
- For diagrams with no direct native representation in the chosen renderer, explain the approximation or switch formats. Never hide a missing semantic behind a visually plausible shape.

## Source basis

Visual Paradigm, “The Complete Guide to UML Diagram Types: 14 Visual Models for Clearer Systems,” accessed 2026-07-13:
https://www.visual-paradigm.com/guide/the-complete-guide-to-uml-diagrams-all-14-types-explained-with-practical-examples/
