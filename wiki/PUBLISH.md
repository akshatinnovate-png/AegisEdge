# Publishing these pages to the GitHub wiki

These are the AegisEdge wiki pages, kept in the repository so they are version
controlled and reviewed like everything else. GitHub serves a wiki from a
*separate* git repository, which this session's network policy will not
authorise a push to, so publishing is one command from a machine that can reach
GitHub normally.

```bash
git clone https://github.com/akshatinnovate-png/AegisEdge.wiki.git /tmp/aegis-wiki
cp wiki/*.md /tmp/aegis-wiki/
cd /tmp/aegis-wiki
git add -A && git commit -m "Fill the wiki" && git push
```

`_Sidebar.md` and `_Footer.md` are special filenames GitHub renders as the
navigation sidebar and the page footer. Everything else becomes a page whose
title is the filename with hyphens turned into spaces, which is why the files
are named `Qdrant-at-the-Centre.md` rather than anything friendlier.

## The pages

| File | Page |
|---|---|
| `Home.md` | Landing page, headline evidence, PS mapping |
| `Quick-Start.md` | Two terminals, and the commands that prove it |
| `Architecture.md` | System shape, module map, storage layers |
| `Qdrant-at-the-Centre.md` | The schema, the single call, the bake-off, the router |
| `Hybrid-Retrieval.md` | Dense, sparse, fusion, rerank, scoring, the index |
| `Sync-Conflicts-and-Egress.md` | CRDT, Merkle, IBLT, conflicts, value-per-byte egress |
| `Security-Model.md` | Identity, signing, admission, enrolment, tenancy, audit |
| `Reliability-and-Degradation.md` | WAL, segments, the SLO ladder, live invariants, chaos |
| `How-This-Was-Tested.md` | Simulation, the determinism lint, bake-offs, the claims audit, CI |
| `Defect-Log.md` | The 35 defects, with the ones worth knowing about written out |
| `Open-Problems.md` | Four things still wrong, and three bounded claims |
| `API-Reference.md` | Every endpoint, and what a search response carries |
| `Configuration.md` | Every `AEGIS_*` variable that matters |
| `Console-Guide.md` | What each panel shows and the rule they all follow |
| `FAQ.md` | The questions a judge would ask |
