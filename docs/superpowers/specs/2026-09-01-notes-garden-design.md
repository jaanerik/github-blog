# Design: Obsidian-style digital garden at jaanerikpihel.com/notes

Date: 2026-09-01
Status: approved

## Goal

Add a Quartz 4 digital garden (wikilinks, backlinks, graph view, explorer,
fuzzy search, hover previews) served at `jaanerikpihel.com/notes`, while the
existing Jekyll/minimal-mistakes blog stays unchanged at the site root.
Notes are authored in nvim as Markdown; posts needing executed Python are
authored as Quarto `.qmd` and executed at build time with caching.

## Non-goals

- No migration or restyling of the 4 existing Jekyll posts.
- No change to the blog theme, domain, or URLs.
- No Python execution in CI (execution happens locally; results committed).

## Repo layout (additions)

```
notes/               # the vault: .md notes + .qmd executable posts
  index.md           # garden home page
_quarto.yml          # Quarto project config (freeze: auto, output gfm)
_freeze/             # committed Quarto execution results
quartz/              # Quartz 4 generator (vendored from upstream template)
quartz.config.ts     # Quartz config: baseUrl, theme, KaTeX macros, plugins
.github/workflows/deploy.yml
```

`notes/`, `_freeze/`, `_quarto.yml`, `quartz/`, `quartz.config.ts`, and
Quartz build/node dirs are added to Jekyll `exclude:` in `_config.yml`.

## Authoring flow

- Plain notes: `.md` with wikilinks `[[note-name]]`; YAML frontmatter
  (`title`, optional `tags`, `draft`).
- Executable posts: `.qmd` with Python code blocks. Run `quarto render`
  locally; outputs become Markdown + figure files that Quartz treats as
  ordinary notes.
- Caching: `freeze: auto` — a `.qmd` re-executes only when its own source
  changes. `_freeze/` is committed, so CI never executes Python.
- LaTeX: KaTeX via Quartz. Custom macros (e.g. `\indep`) defined once in the
  KaTeX macros block of `quartz.config.ts`.

## Site features

Quartz defaults, all enabled: explorer sidebar, fuzzy search (Ctrl+K),
interactive graph view, per-note backlinks, hover-preview popovers,
dark/light toggle defaulting dark. Palette nudged toward Obsidian in
`quartz/styles/custom.scss`. `baseUrl: "jaanerikpihel.com/notes"` so links,
search, sitemap, and graph resolve under the subpath.

## Blog integration

Add a "Notes" link to the blog masthead via `_data/navigation.yml`
(`main:` entry pointing to `/notes/`).

## Deploy

Pages source flips from "deploy from branch" to "GitHub Actions" (one-time
manual settings change). One workflow on push to main:

1. Build Jekyll site (existing Gemfile) into `_site/`.
2. Install Quarto; `quarto render` (thaws `_freeze/`, no Python env).
3. `npx quartz build` on the rendered notes.
4. Copy Quartz output into `_site/notes/`.
5. Upload merged artifact; deploy to GitHub Pages. CNAME preserved.

Trade-off: the automatic branch build goes away; a broken workflow blocks
site updates but fails visibly (red X) instead of silently.

## Verification

- Local preview: `quarto render && npx quartz build --serve` plus
  `bundle exec jekyll serve` for the blog.
- Checklist: KaTeX macros render; wikilinks/graph resolve under `/notes`;
  a `.qmd` post shows code + executed output; editing an unrelated file
  does not re-execute other posts; blog pages byte-identical in intent
  (same URLs, theme); masthead Notes link works; CI deploy green on the
  custom domain.
