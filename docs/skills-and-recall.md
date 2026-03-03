# Skills and Session Recall

Prscope now supports a layered planning memory model that compounds over time while keeping behavior deterministic.

## Memory Layers

1. **Manifesto** (`.prscope/manifesto.md`)  
   Project philosophy and hard/soft constraints.
2. **Skills** (`.prscope/skills/*.md`)  
   Stable team defaults and planning patterns.
3. **Recall** (`prscope recall ...`)  
   Episodic memory from prior planning sessions.
4. **Memory blocks** (`architecture`, `modules`, `patterns`, `entrypoints`)  
   Working codebase context for the current repository state.

Injection order is intentionally fixed:

1. Manifesto
2. Skills
3. Recall
4. Memory blocks

This ensures historical precedent cannot override policy and constraints.

## Skills

Skills are markdown files loaded from `.prscope/skills/`.

- Deterministic alphabetical ordering (`sorted(glob("*.md"))`)
- UTF-8 file reads
- Boundary-safe truncation (never mid-file)
- Explicit startup logging for loaded/trimmed/skipped files

Create skill files to encode team planning bias such as migration discipline, API compatibility checks, or test expectations.

Example:

```bash
mkdir -p .prscope/skills
$EDITOR .prscope/skills/api.md
```

## Recall

Recall is powered by BM25 over historical planning sessions in SQLite.

- Corpus: title + requirements + latest saved plan content
- Recency boost: newer sessions rank higher
- Default scope: current repo
- Cross-repo search: `--all-repos`
- Raw output mode: `--full`

Examples:

```bash
prscope recall "auth token rotation endpoint compatibility"
prscope recall "auth token rotation endpoint compatibility" --all-repos
prscope recall "auth token rotation endpoint compatibility" --full
```

## Clean-Slate Planning

Disable recall for a session when you want a fresh plan without precedent anchoring:

```bash
prscope plan start --no-recall "Design a new auth subsystem"
prscope plan chat --no-recall
```

`--no-recall` is persisted with the planning session.
