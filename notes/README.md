# Notes garden authoring

- Plain notes: `.md` files with `[[wikilinks]]`.
- Executable posts: `.qmd` files under `posts/`, using standard `[markdown](links)` — wikilinks get pandoc-escaped.
- Local render (needed for `.qmd`, PEP 668 blocks system pip):
  `source .venv/bin/activate && quarto render notes`
  (venv has jupyter + matplotlib for code execution.)
- Preview: `cd quartz && npx quartz build -d ../notes -o ../_site/notes --serve` (localhost:8080/).
- Commit sources + rendered `.md` + `_freeze/` together.
- Rendered `.md` files are generated — edit the `.qmd`, not them.
