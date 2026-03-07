# Frontend Architecture

This document covers the React/Vite frontend in `src/prscope/web/frontend/`.

## Stack

- **React 18** with TypeScript
- **Vite** dev server + production build
- **TanStack Query** for server state
- **React Router** for client-side routing
- **Tailwind CSS** for styling
- **SSE** (Server-Sent Events) for real-time updates
- **Lucide React** for icons

## Directory Layout

Layout under `src/prscope/web/frontend/src/`:

```
src/
├── main.tsx                    ← app entry point, React Router setup
├── App.tsx                     ← root route layout
├── index.css                   ← Tailwind base + custom theme variables
├── types.ts                    ← shared TypeScript types
├── pages/
│   ├── SessionList.tsx         ← session list / home page
│   ├── NewSession.tsx          ← session creation form
│   ├── PlanningView.tsx        ← main planning workspace (chat + plan + issues)
│   └── PlanningView.test.ts   ← tests for timeline reducer, buildTimeline, upsertToolCall
├── components/
│   ├── ActionBar.tsx           ← session action buttons (round, approve, export)
│   ├── ChatPanel.tsx           ← pure timeline renderer (chat + tools)
│   ├── ChatPanel.test.ts      ← tests for hasRunningToolCalls
│   ├── chatPanelUtils.ts      ← timeline reducer, buildTimeline, upsertToolCall utilities
│   ├── PlanPanel.tsx           ← rendered plan with diff view
│   ├── IssuePanel.tsx          ← issue graph / open issues view
│   ├── ResizableLayout.tsx     ← draggable split pane
│   ├── ToolCallStream.tsx      ← live tool call display
│   ├── ModelSelector.tsx       ← model picker dropdown
│   ├── OptionButtons.tsx       ← multi-choice response buttons
│   ├── CritiqueCard.tsx        ← critique display card
│   ├── ThemeToggle.tsx         ← dark/light mode toggle
│   └── ui/
│       └── Tooltip.tsx         ← reusable tooltip primitive
├── hooks/
│   ├── useSessionEvents.ts    ← SSE connection + event normalization
│   ├── useTheme.ts            ← theme preference persistence
│   └── useMediaQuery.ts       ← responsive breakpoint hook
└── lib/
    ├── api.ts                 ← HTTP client (fetch wrappers for all endpoints)
    ├── markdown.ts            ← markdown rendering utilities
    ├── markdownComponents.tsx ← custom markdown component overrides
    ├── decisionGraphRender.ts ← graph-backed plan augmentation helpers
    └── planTitle.ts           ← plan title extraction/cleaning
```

## Key Patterns

### Server-Authoritative State

The frontend treats the server as the source of truth:

- `GET /api/sessions/{id}` is sufficient to render any screen.
- `session_state` SSE events replace local state entirely (not merged).
- On reconnect, the first SSE event is always a full `session_state` snapshot.
- Session snapshot payloads can include both legacy `open_issues` and additive `issue_graph` fields.

**Rule:** Never reconstruct UI state from turns, events, or local cache. Always derive from the latest server snapshot.

### SSE Event Handling

`useSessionEvents` manages the SSE connection lifecycle:

- Connects to `GET /api/sessions/{id}/events`
- Normalizes raw SSE events into typed `UIEvent` objects
- Extracts `session_version` from every event payload for ordering guarantees
- `session_state` events are treated as total state replacement
- `tool_update` events carry a unified `ToolCallEntry` with `call_id` and `status` (`running` | `done`)
- Connection drops trigger automatic reconnection

### Timeline Architecture

The frontend renders chat history and tool execution as a single chronological timeline.

**State ownership:** A `timelineReducer` in `PlanningView` owns the timeline via `useReducer`. It stores `turns`, `groups`, the merged `timeline`, and `activeTools` in a single state object.

**Timeline projection:** `buildTimeline()` performs an O(n) linear merge of turns and completed tool groups, sorted by monotonic `sequence` numbers assigned server-side. Tool groups with the same sequence as a turn sort first (tools appear before the response they produced).

**Active tools** are transient UI state, rendered separately below the timeline — they are not timeline items. This prevents visual flicker when tools complete and move into a finalized group.

**Event version gating:** Every SSE event carries a monotonic `session_version`. The frontend tracks `lastVersionSeen` and discards any event with a version ≤ the last seen, preventing stale/reordered events from causing UI glitches.

**Data flow:**

```
SSE event → version gate → dispatch to timelineReducer
                              ↓
                        TimelineState { timeline, activeTools }
                              ↓
                        ChatPanel (pure renderer)
```

### Command Model

All mutating actions go through `POST /api/sessions/{id}/command` (or wrapper endpoints). Each command carries a `command_id` UUID for idempotent replay. The server rejects concurrent commands with `409`.

The frontend handles `409` responses gracefully — they indicate the session is busy, not an error.

### Component Responsibilities

| Component | Owns | Does Not Own |
|---|---|---|
| `PlanningView` | Session lifecycle, SSE subscription, timeline reducer, layout | Individual panel rendering |
| `ChatPanel` | Pure timeline rendering, message input, discovery flow | State management, timeline construction |
| `chatPanelUtils` | `timelineReducer`, `buildTimeline`, `upsertToolCall` | Component rendering |
| `PlanPanel` | Plan rendering, diff view, export | Round execution |
| `IssuePanel` | Issue graph / open issues display | State management |
| `ActionBar` | Command dispatch (round, approve, export) | State management |
| `ToolCallStream` | Tool call display and filtering | Tool execution |

### Issue Graph Rendering

- When `snapshot.issue_graph` is present, `PlanPanel` renders a causal issue tree from `causes` edges.
- `depends_on` and duplicate aliases are rendered as annotations, not as tree edges.
- When `issue_graph` is absent, UI falls back to legacy flat issue counters from `open_issues`.
- Round tables remain backward-compatible (`major_issues`, `minor_issues`) with optional graph summaries.

### Decision Graph Rendering

- `PlanPanel` receives `current_plan.decision_graph` from the backend and augments markdown before rendering.
- `decisionGraphRender.ts` injects graph-backed architecture decisions and unresolved questions that may not yet be reflected in prose.
- Follow-up prompts rendered in `ChatPanel` come from persisted follow-up artifacts derived from the same decision graph.
- The frontend never infers or mutates decision state locally; graph state is server-authoritative just like session state.

### Styling

- Tailwind utility classes for all styling.
- CSS custom properties in `index.css` for theme tokens.
- Dark mode via `useTheme` hook + `dark` class on root element.
- No CSS modules, no styled-components.

## Build and Lint

```bash
cd src/prscope/web/frontend
npm ci               # install deps
npm run dev           # vite dev server on :5173
npm run build         # production build → src/prscope/web/static/
npm run lint          # eslint
npx tsc --noEmit      # typecheck
```

Production builds are committed to `src/prscope/web/static/` for the Python server to serve directly.

## Adding a New Component

1. Create the file in the appropriate directory (`components/`, `pages/`, `hooks/`, `lib/`).
2. Keep components focused — one responsibility per file.
3. Use TypeScript types from `types.ts` for shared data shapes.
4. If the component needs server data, use TanStack Query or the SSE hook — never raw fetch.
5. Run `npm run lint && npx tsc --noEmit` before committing.

## Related Documents

- [Agent Harness](./agent-harness.md) — SSE contract and API surface
- [Planning State Machine](./planning-state-machine.md) — valid states and transitions
- [Design Philosophy](./DESIGN.md) — why server-authoritative
