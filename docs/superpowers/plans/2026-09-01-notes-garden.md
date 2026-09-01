# Notes Garden Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a Quartz 4 digital garden (wikilinks, backlinks, graph, search) at jaanerikpihel.com/notes alongside the untouched Jekyll blog at the root, with Python in `.qmd` posts executed locally via Quarto freeze.

**Architecture:** Quartz 4 is vendored into `quartz/`; the vault lives in `notes/`, which is also a Quarto project (`notes/_quarto.yml`) rendering `.qmd → .md` in place with `freeze: auto`. One GitHub Actions workflow builds Jekyll into `_site/`, thaws/renders Quarto, builds Quartz into `_site/notes/`, and deploys the merged artifact to Pages.

**Tech Stack:** Jekyll (github-pages gem, minimal-mistakes remote theme), Quartz 4 (Node ≥20, installed: v25), Quarto 1.9, Python 3.14 + Jupyter for local execution only.

**Spec:** `docs/superpowers/specs/2026-09-01-notes-garden-design.md`

## Global Constraints

- Blog URLs, theme, and posts must not change; Jekyll config changes are limited to `exclude:` additions and `_data/navigation.yml`.
- Quartz `baseUrl` is exactly `jaanerikpihel.com/notes`.
- CI never executes Python: `_freeze/` (at `notes/_freeze/`) and rendered `.md` outputs are committed.
- Custom KaTeX macro `\indep` → `\perp\!\!\!\perp` must render in notes.
- Deviations from spec (agreed rationale): `_quarto.yml` and `_freeze/` live inside `notes/` (a root `_quarto.yml` would make the whole repo a Quarto project and render `README.md`/`about.md`); rendered `.md` + `*_files/` are committed so a forgotten local render fails loudly in CI rather than silently.
- All work happens on `main` (single-author blog repo; user has been committing to main).

---

### Task 1: Vendor Quartz and configure it for the /notes subpath

**Files:**
- Create: `quartz/` (vendored clone of github.com/jackyzha0/quartz, v4, with its own nested `.gitignore` kept)
- Modify: `quartz/quartz.config.ts`
- Modify: `quartz/quartz/styles/custom.scss`

**Interfaces:**
- Produces: a working `npx quartz build -d <contentdir> -o <outdir>` invocation run from inside `quartz/`; Quartz ignores `.qmd`, `_quarto.yml`, `_freeze`. Task 3 and Task 5 rely on exactly `npx quartz build -d ../notes -o ../_site/notes`.

- [ ] **Step 1: Vendor Quartz**

```bash
cd /Users/erik/github-blog
git clone --depth 1 https://github.com/jackyzha0/quartz.git quartz
rm -rf quartz/.git
cd quartz && npm ci
```

Expected: install completes without errors (warnings OK). Note: `quartz/.gitignore` from upstream already ignores `node_modules` and `public`.

- [ ] **Step 2: Configure quartz.config.ts**

In `quartz/quartz.config.ts`, edit the existing fields (do not rewrite the file; upstream structure must be preserved):

- `pageTitle: "Erik's notes"`
- `baseUrl: "jaanerikpihel.com/notes"`
- `ignorePatterns: ["private", "templates", ".obsidian", "**/*.qmd", "_quarto.yml", "_freeze"]`
- In `plugins.transformers`, find `Plugin.Latex({ renderEngine: "katex" })` and change to:

```ts
Plugin.Latex({
  renderEngine: "katex",
  customMacros: {
    "\\indep": "\\perp\\!\\!\\!\\perp",
    "\\E": "\\mathbb{E}",
    "\\R": "\\mathbb{R}",
  },
}),
```

If the installed Quartz version's Latex plugin has no `customMacros` option (check `quartz/quartz/plugins/transformers/latex.ts` for the option name — older versions call it `customMacros`, do not guess others), fall back to editing that plugin file to pass `macros` into the katex options, and note the deviation in the commit message.

- [ ] **Step 3: Verify build with placeholder content**

```bash
cd /Users/erik/github-blog
mkdir -p notes
printf -- '---\ntitle: Home\n---\nWelcome. See [[test-note]]. Math: $x \\indep y$.\n' > notes/index.md
printf -- '---\ntitle: Test note\n---\nBacklink target. $\\E[X] \\in \\R$.\n' > notes/test-note.md
cd quartz && npx quartz build -d ../notes -o ../_site/notes
```

Expected: build succeeds; `_site/notes/index.html` and `_site/notes/test-note.html` exist.

- [ ] **Step 4: Verify subpath links and macros in output**

```bash
grep -o 'href="[^"]*test-note[^"]*"' ../_site/notes/index.html | head -3
grep -c 'katex' ../_site/notes/index.html
grep -c 'indep' ../_site/notes/index.html || true
```

Expected: the wikilink resolved to an href (relative or subpath-absolute, NOT pointing to domain root without `/notes`); katex markup present. If the literal string `\indep` appears in rendered math HTML as an unknown-macro error (katex renders unknown macros in red with class `katex-error`), the macro config is wrong — fix before proceeding. Check: `grep -c 'katex-error' ../_site/notes/index.html` should print `0`.

- [ ] **Step 5: Obsidian-dark styling nudge**

In `quartz/quartz/styles/custom.scss`, append:

```scss
// Obsidian-ish accent tweaks
:root[saved-theme="dark"] {
  --secondary: #a882ff; // Obsidian purple accent
  --tertiary: #7c5cd6;
}
```

Then verify variable names against `quartz/quartz/styles/themes` / `quartz.config.ts` `colors` block: Quartz defines its palette in `quartz.config.ts` under `theme.colors.darkMode` (keys: `light`, `lightgray`, `gray`, `darkgray`, `dark`, `secondary`, `tertiary`, `highlight`). Preferred: set `theme.colors.darkMode.secondary = "#a882ff"` and `tertiary = "#7c5cd6"` in `quartz.config.ts` instead of the SCSS override, and delete the SCSS block; leave all other colors at defaults. Rebuild (`npx quartz build -d ../notes -o ../_site/notes`) and confirm no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/erik/github-blog
git add quartz notes
git commit -m "Vendor Quartz 4 configured for /notes subpath"
```

Note: `git add quartz` must pick up the vendored files as regular tracked files (no submodule). `git ls-files quartz | head` should list many files; if it prints a single `quartz` gitlink entry, the `rm -rf quartz/.git` step was missed.

---

### Task 2: Quarto project in notes/ with executable sample post

**Files:**
- Create: `notes/_quarto.yml`
- Create: `notes/posts/sample-qmd-post.qmd`
- Create (generated, committed): `notes/posts/sample-qmd-post.md`, `notes/posts/sample-qmd-post_files/`, `notes/_freeze/`

**Interfaces:**
- Consumes: nothing from Task 1 (independent of Quartz).
- Produces: `quarto render notes` as the canonical render command (run from repo root); rendered `.md` files next to their `.qmd` sources. Tasks 3 and 5 rely on `quarto render notes` exactly.

- [ ] **Step 1: Ensure a Jupyter kernel exists locally**

```bash
python3 -m pip show jupyter matplotlib 2>/dev/null | grep -E '^(Name|Version)' || python3 -m pip install jupyter matplotlib
```

Expected: jupyter and matplotlib available (install if missing; if `pip install` is blocked by PEP 668 externally-managed environment, use `python3 -m pip install --user jupyter matplotlib` or an existing venv — ask the user which env they use for Python work before creating anything new).

- [ ] **Step 2: Write notes/_quarto.yml**

```yaml
project:
  render:
    - "**/*.qmd"

execute:
  freeze: auto

format:
  gfm:
    output-ext: md
```

- [ ] **Step 3: Write the sample executable post**

`notes/posts/sample-qmd-post.qmd`:

````markdown
---
title: "Sample executed post"
format: gfm
---

Links back to [[test-note]].

The identity $\E[X+Y] = \E[X] + \E[Y]$ holds even when $X \not\indep Y$.

```{python}
#| label: fig-sine
#| fig-cap: "A sine wave, executed at render time"
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 200)
plt.plot(x, np.sin(x))
plt.show()
```

```{python}
print(f"computed at render time: {np.trapezoid(np.sin(x), x):.4f}")
```
````

- [ ] **Step 4: Render and verify execution**

```bash
cd /Users/erik/github-blog && quarto render notes
```

Expected: renders `sample-qmd-post.qmd` (executes Python), creates `notes/posts/sample-qmd-post.md`, a figure under `notes/posts/sample-qmd-post_files/`, and `notes/_freeze/posts/sample-qmd-post/`. The two plain `.md` notes from Task 1 must NOT be touched (the `render:` list restricts to `*.qmd`).

- [ ] **Step 5: Verify math survives as raw TeX**

```bash
grep -n 'indep\|\\E\[' notes/posts/sample-qmd-post.md | head -5
```

Expected: `$...$`-delimited TeX with `\indep`/`\E` intact in the output markdown. If instead math appears as `latex.codecogs.com`/webtex image URLs, add `html-math-method: katex` under the `gfm:` block in `notes/_quarto.yml` and re-render; verify again.

- [ ] **Step 6: Verify freeze caching**

```bash
touch notes/index.md && quarto render notes 2>&1 | tee /tmp/render2.log
grep -ci 'executing\|starting python3 kernel' /tmp/render2.log || echo "no execution"
```

Expected: second render does not start a Python kernel (output shows the qmd read from freeze or skipped entirely). If it re-executes, `freeze: auto` is misconfigured — stop and fix.

- [ ] **Step 7: Commit (including generated outputs)**

```bash
git add notes
git commit -m "Add Quarto project with frozen executable sample post"
```

---

### Task 3: Full local pipeline verification

**Files:**
- Modify: `quartz/quartz.config.ts` only if Step 2 finds `.qmd`/`_freeze` leakage.

**Interfaces:**
- Consumes: `npx quartz build -d ../notes -o ../_site/notes` (Task 1), rendered notes (Task 2).
- Produces: confirmation the two-stage build is sound; no new artifacts.

- [ ] **Step 1: Build Quartz over the real vault**

```bash
cd /Users/erik/github-blog/quartz && npx quartz build -d ../notes -o ../_site/notes
```

Expected: success; `_site/notes/posts/sample-qmd-post.html` exists.

- [ ] **Step 2: Verify no build-internals leaked into the site**

```bash
ls ../_site/notes | head -20
find ../_site/notes -name '*.qmd' -o -name '_quarto*' | wc -l
```

Expected: `0` leaked files; no `_freeze` directory in output.

- [ ] **Step 3: Verify the executed post rendered fully**

```bash
grep -c 'img' ../_site/notes/posts/sample-qmd-post.html
grep -c 'computed at render time' ../_site/notes/posts/sample-qmd-post.html
grep -c 'katex-error' ../_site/notes/posts/sample-qmd-post.html
```

Expected: ≥1 img (the sine figure — also verify the figure file was copied into `_site/notes/` output, not just referenced), 1 for the printed output, 0 katex errors.

- [ ] **Step 4: Interactive smoke test**

```bash
npx quartz build -d ../notes -o ../_site/notes --serve
```

Open http://localhost:8080 — note: with a subpath baseUrl Quartz serves under `/notes` locally too; try http://localhost:8080/notes if root 404s. Check: explorer sidebar lists notes; graph view shows index ↔ test-note ↔ sample post links; search (Ctrl+K) finds "sine"; backlinks panel on test-note lists the sample post; `\indep` renders as ⊥⊥. Then Ctrl-C the server. This is a human-check step — report what was seen.

- [ ] **Step 5: Commit (only if config changed)**

```bash
git status --short
# if quartz.config.ts changed:
git add quartz/quartz.config.ts && git commit -m "Tighten Quartz ignore patterns"
```

---

### Task 4: Jekyll integration — excludes and Notes masthead link

**Files:**
- Modify: `_config.yml`
- Create: `_data/navigation.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: Jekyll build that ignores `notes/`, `quartz/`, `docs/`; masthead "Notes" link to `/notes/`.

- [ ] **Step 1: Add excludes to _config.yml**

Append to `_config.yml` (the file's `exclude:` block is currently commented out; add a live one):

```yaml
exclude:
  - notes/
  - quartz/
  - docs/
  - vendor/
  - Gemfile
  - Gemfile.lock
```

(Explicit `exclude:` replaces Jekyll's default list, so re-list `vendor/`, `Gemfile`, `Gemfile.lock` which the defaults normally cover.)

- [ ] **Step 2: Create _data/navigation.yml**

```yaml
main:
  - title: "Notes"
    url: /notes/
```

- [ ] **Step 3: Build Jekyll and verify blog unchanged + notes excluded**

```bash
cd /Users/erik/github-blog && bundle install && bundle exec jekyll build 2>&1 | tail -5
ls _site/notes 2>/dev/null && echo "LEAK: jekyll copied notes" || echo "OK: notes excluded"
grep -c 'href="/notes/"' _site/index.html
ls _site | grep -E 'about|feed.xml|2025' >/dev/null && echo "blog pages present"
```

Expected: build succeeds; "OK: notes excluded" (note: Jekyll build wipes `_site/`, so the Quartz output from Task 3 disappearing here is expected — build order matters, Jekyll first); ≥1 Notes link in the homepage masthead; existing blog pages still generated. Caveat: `bundle exec jekyll build` with `remote_theme` needs network access; if the github-pages gem fails locally on Ruby version issues, note it and rely on CI verification in Task 5 — but still verify `_config.yml` syntax with `ruby -ryaml -e 'YAML.load_file("_config.yml")' && echo valid`.

- [ ] **Step 4: Commit**

```bash
git add _config.yml _data/navigation.yml
git commit -m "Exclude garden dirs from Jekyll; add Notes masthead link"
```

---

### Task 5: Deploy workflow and Pages cutover

**Files:**
- Create: `.github/workflows/deploy.yml`

**Interfaces:**
- Consumes: `quarto render notes` (Task 2), `npx quartz build -d ../notes -o ../_site/notes` (Task 1), Jekyll build (Task 4).
- Produces: the live merged site.

- [ ] **Step 1: Write .github/workflows/deploy.yml**

```yaml
name: Deploy blog + notes

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: ruby/setup-ruby@v1
        with:
          ruby-version: "3.3"
          bundler-cache: true

      - name: Build Jekyll site
        run: bundle exec jekyll build
        env:
          JEKYLL_ENV: production

      - uses: quarto-dev/quarto-actions/setup@v2
        with:
          version: "1.9.38"

      - name: Render notes (thaw freeze, no execution)
        run: quarto render notes

      - uses: actions/setup-node@v4
        with:
          node-version: 22

      - name: Build Quartz into _site/notes
        working-directory: quartz
        run: |
          npm ci
          npx quartz build -d ../notes -o ../_site/notes

      - uses: actions/upload-pages-artifact@v3
        with:
          path: _site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/deploy.yml
git commit -m "Add merged Jekyll + Quartz Pages deploy workflow"
git push origin main
```

- [ ] **Step 3: USER ACTION — flip Pages source**

Ask the user to open https://github.com/jaanerik/github-blog/settings/pages and change **Source** from "Deploy from a branch" to "**GitHub Actions**". The workflow pushed in Step 2 may fail or the branch build may race until this is flipped — that's expected. Wait for the user's confirmation before Step 4.

- [ ] **Step 4: Verify the workflow run**

```bash
gh run watch --repo jaanerik/github-blog $(gh run list --repo jaanerik/github-blog --workflow "Deploy blog + notes" --limit 1 --json databaseId --jq '.[0].databaseId')
```

Expected: run completes green. If the Ruby/github-pages step fails on Ruby 3.3, retry with `ruby-version: "3.2"` (the github-pages gem pins old Jekyll; 3.2 is the safe floor) — commit the fix and re-push.

- [ ] **Step 5: Verify the live site**

```bash
curl -sI https://jaanerikpihel.com/ | head -3
curl -s https://jaanerikpihel.com/ | grep -c 'href="/notes/"'
curl -sI https://jaanerikpihel.com/notes/ | head -3
curl -s https://jaanerikpihel.com/notes/posts/sample-qmd-post.html | grep -c 'computed at render time'
```

Expected: blog 200 with Notes link; /notes/ 200; executed output present. Custom domain + HTTPS should carry over automatically (the domain is set in Pages settings, not the CNAME file, for Actions deploys — if /notes/ 404s but the run was green, re-check the Pages source setting and that "Custom domain" still shows jaanerikpihel.com).

- [ ] **Step 6: Confirm blog regression-free**

```bash
for p in / /about/ /2025/08/23/intro.html; do curl -s -o /dev/null -w "%{http_code} $p\n" https://jaanerikpihel.com$p; done
```

Expected: 200s on / and /about/. For the post URL: first check the real permalink pattern in the live site's homepage HTML (minimal-mistakes default permalinks may differ from this guess) — verify whatever post URLs the homepage links to return 200, comparing against the pre-cutover site structure, not against this guessed path.

---

## Post-plan notes (not tasks)

- The 4 existing blog posts are untouched by design.
- The GitHub-Skills tutorial workflows (`0-welcome.yml` … `5-merge-your-pull-request.yml`) are inert but could be deleted later; out of scope.
- Authoring loop going forward: edit in nvim → `quarto render notes` (only changed `.qmd` re-executes) → `cd quartz && npx quartz build -d ../notes -o ../_site/notes --serve` to preview → commit `notes/` (sources + rendered md + `_freeze/`) → push.
