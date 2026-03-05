# Design Docs Index

Catalog of design documents for significant architectural and product decisions.

## Format

Each design doc should include:

- **Status**: `draft` | `accepted` | `superseded` | `rejected`
- **Date**: when the decision was made
- **Context**: what problem or opportunity prompted this
- **Decision**: what was decided and why
- **Consequences**: known tradeoffs and follow-up work

## Catalog

| Doc | Status | Date | Summary |
|---|---|---|---|
| (none yet) | — | — | — |

## Creating a New Design Doc

1. Create a file in this directory: `docs/design-docs/NNNN-short-title.md`
2. Use sequential numbering (next available `NNNN`).
3. Include the fields listed above.
4. Add an entry to the catalog table.
5. Link from relevant code or docs if the decision affects behavior.

Design docs are versioned in git. They are not living documents — once accepted, they capture the decision at that point in time. If the decision changes, create a new doc and mark the old one `superseded`.
