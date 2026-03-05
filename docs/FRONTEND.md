# Frontend Architecture

This document covers the React/Vite frontend in `prscope/web/frontend/`.

## Stack

- **React 18** with TypeScript
- **Vite** dev server + production build
- **TanStack Query** for server state
- **React Router** for client-side routing
- **Tailwind CSS** for styling
- **SSE** (Server-Sent Events) for real-time updates
- **Lucide React** for icons

## Directory Layout

```
src/
├── main.tsx                    ← app entry point, React Router setup
├── App.tsx                     ← root route layout
├── index.css                   ← Tailwind base + custom theme variables
├── types.ts                    ← shared TypeScript types
├── pages/
│   ├── SessionList.tsx         ← session list / home page
│   ├── NewSession.tsx          ← session creation form
│   └── PlanningView.tsx        ← main planning workspace (chat + plan)
├── components/
│   ├── ActionBar.tsx           ← session action buttons (round, approve, export)
│   ├── ChatPanel.tsx           ← discovery chat / message history
│   ├── PlanPanel.tsx           ← rendered plan with diff view
│   ├── ResizableLayout.tsx     ← draggable split pane
│   ├── ToolCallStream.tsx      ← live tool call display
│   ├── ModelSelector.tsx       ← model picker dropdown
│   ├── OptionButtons.tsx       ← multi-choice response buttons
│   ├── CritiqueCard.tsx        ← critique display card
│   ├── ThemeToggle.tsx         ← dark/light mode toggle
│   └── ui/
│       └── Tooltip.tsx         ← reusable tooltip primitive
├── hooks/
│   ├── useSessionEvents.ts    ← SSE connection + event dispatch
│   ├── useTheme.ts            ← theme preference persistence
│   └── useMediaQuery.ts       ← responsive breakpoint hook
└── lib/
    ├── api.ts                 ← HTTP client (fetch wrappers for all endpoints)
    ├── markdown.ts            ← markdown rendering utilities
    ├── markdownComponents.tsx ← custom markdown component overrides
    └── planTitle.ts           ← plan title extraction/cleaning
```

## Key Patterns

### Server-Authoritative State

The frontend treats the server as the source of truth:

- `GET /api/sessions/{id}` is sufficient to render any screen.
- `session_state` SSE events replace local state entirely (not merged).
- On reconnect, the first SSE event is always a full `session_state` snapshot.

**Rule:** Never reconstruct UI state from turns, events, or local cache. Always derive from the latest server snapshot.

### SSE Event Handling

`useSessionEvents` manages the SSE connection lifecycle:

- Connects to `GET /api/sessions/{id}/events`
- Dispatches typed events to callbacks
- `session_state` events are treated as total state replacement
- Tool call events (`tool_call`, `tool_result`) update an in-memory list
- Connection drops trigger automatic reconnection

### Command Model

All mutating actions go through `POST /api/sessions/{id}/command` (or wrapper endpoints). Each command carries a `command_id` UUID for idempotent replay. The server rejects concurrent commands with `409`.

The frontend handles `409` responses gracefully — they indicate the session is busy, not an error.

### Component Responsibilities

| Component | Owns | Does Not Own |
|---|---|---|
| `PlanningView` | Session lifecycle, SSE subscription, layout | Individual panel rendering |
| `ChatPanel` | Message display, input, discovery flow | Session state mutations |
| `PlanPanel` | Plan rendering, diff view, export | Round execution |
| `ActionBar` | Command dispatch (round, approve, export) | State management |
| `ToolCallStream` | Tool call display and filtering | Tool execution |

### Styling

- Tailwind utility classes for all styling.
- CSS custom properties in `index.css` for theme tokens.
- Dark mode via `useTheme` hook + `dark` class on root element.
- No CSS modules, no styled-components.

## Build and Lint

```bash
cd prscope/web/frontend
npm ci               # install deps
npm run dev           # vite dev server on :5173
npm run build         # production build → prscope/web/static/
npm run lint          # eslint
npx tsc --noEmit      # typecheck
```

Production builds are committed to `prscope/web/static/` for the Python server to serve directly.

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
