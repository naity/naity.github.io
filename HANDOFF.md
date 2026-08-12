# Redesign handoff

Scratch note for a session picking up the `naity.github.io` redesign. **Delete this
file before merging to `main`.**

## What this is

A ground-up rebuild of Yuan Tian's personal site (GitHub Pages, https://naity.github.io).
The repo used to contain only the *built output* of an old Hugo/LoveIt blog; it is now a
real **Astro 7 + Tailwind v4** source project that builds from `src/` and deploys via
GitHub Actions. Clean-minimal design, light/dark, zinc neutrals + teal accent, self-hosted
Inter, near-zero client JS.

- **Work branch:** `claude/personal-page-redesign-blyr75` (push here; **no PR unless asked**).
- **Env:** Node 22+ (`node --version`), Astro `^7.2.0`, Tailwind `^4.3.3`, `sharp` for images.
- **Commands:** `npm install`, `npm run dev` (localhost:4321), `npm run build`, `npm run preview`, `npm run check` (astro check).

## Status: done (Medium curation completed 2026-08-12)

The Medium curation task below was completed: 5 external articles are live in
`src/data/externalPosts.ts` (resume matcher with confirmed 2025-12-14 date, tool-using LLM
agent, protein transformers, protein science primer, TCR specificity). The other top picks
(image+tabular, single-cell multi-omics, ReFocus, UFC ChatGPT) already exist as native posts,
so they were not added as link-outs. No multi-part series found for the resume matcher yet.
Native migration of Medium posts was deferred (Yuan chose "decide later").

Everything below is built, committed, and pushed:

- **Pages:** Home (hero + about + experience timeline + education + featured projects + recent
  writing), `/projects/`, `/publications/` (single year-grouped page), `/blog/` + `/blog/[slug]/`,
  `/photos/` (dialog lightbox), `/404`, RSS at `/index.xml`, sitemap. Old Hugo URLs redirect;
  publication PDFs keep their old `/publications/<slug>/…pdf` paths.
- **Content migrated:** 4 native blog posts (from the old Hugo site), 20 publications with
  PubMed-verified DOIs (incl. fixing the dengue-review journal, upgrading the ADTnorm preprint to
  its Nature Communications 2025 version, and adding the Blood 2025 myeloma paper), 12 photos
  downscaled ~129 MB → ~4 MB, favicons, Search Console file, robots.txt.
- **Experience/education:** taken from Yuan's LinkedIn PDF — Anthropic (MTS, Jun 2026–present),
  AWS (Applied Scientist GenAI), Gilead, Fred Hutch, La Jolla Institute; education = Georgia Tech
  MS CS (2024–2027 expected), UAB PhD, Tianjin BE. All accurate; no placeholders left here.
- **Deploy:** `.github/workflows/deploy.yml` builds + deploys on push to `main`.

### OPEN TASK — curate Medium articles (needs the browser this session has)

The previous (cloud) session **could not reach medium.com** (network egress block), so this was
left for a browser-enabled session. The blog already merges off-site articles with native posts
via a link-out card system that works — it's just seeded with one article and needs the rest.

**How the mechanism works:**
- `src/data/externalPosts.ts` — array of `{ title, date (YYYY-MM-DD), url, source, description, tags? }`.
- `src/components/ExternalPostCard.astro` — renders a link-out card (source badge + external-arrow, no thumbnail).
- `src/lib/posts.ts` `getWritingFeed()` — merges native blog posts + externalPosts, sorted newest-first;
  used by both `/blog/` and the home "Recent writing" section.

**To do:**
1. Read Yuan's Medium profile (https://medium.com/@yuan_tian) in the browser and list his articles.
2. **Curate with Yuan** — he wants "the best ones." Bias: feature GenAI / agents / LLM / ML-for-biology
   pieces (e.g. the agentic resume matcher); likely skip older hobbyist posts (gaming rig, League of
   Legends) unless he wants the full range. Ask him.
3. Add each chosen article as an entry in `externalPosts.ts`.
4. **Fix the seeded entry:** the one existing entry ("Building an Agentic Resume Matcher…", Towards AI)
   has a **placeholder date `2025-12-14` marked `TODO(yuan)`** — confirm the real publication date, and
   check whether it's part of a multi-part series (add the other parts if so).
5. Optional/his call: fully **migrate** any Medium article as a *native* post instead of a link-out
   (better SEO/control). That needs the full text + images — now possible with browser access. Native
   posts live at `src/content/blog/<slug>/index.md` with colocated images; see the existing 4 for the
   frontmatter shape (`title, date, description, tags, cover`).

## Conventions (match these)

- **No em dashes (—) in site copy.** Yuan dislikes them. Use commas/periods, or a colon. En dashes in
  date ranges (`2011 – 2016`) are fine.
- Content/config lives in typed data files under `src/data/` (site, projects, experience, photos,
  externalPosts) and content collections under `src/content/` (blog, publications). Prefer editing data
  files over hardcoding.
- Tailwind **v4** (no `tailwind.config.js`): tokens + `@custom-variant dark` live in `src/styles/global.css`.
  Don't reach for v3 patterns.
- Verify changes with `npm run build` + `npm run check`, and screenshot with Playwright/preview before
  committing. Commit in logical chunks; push to the work branch with `git push -u origin <branch>`.

## After the site is finalized (Yuan does these)

- Merge branch → `main`, then set **repo Settings → Pages → Source: "GitHub Actions"** (site previously
  deployed straight from the branch). Optionally resubmit `sitemap-index.xml` in Search Console.
- `src/data/projects.ts` star counts are a manual snapshot; refresh occasionally. `FM4Life`, `finetune-esm`,
  `image_tabular` are the three `featured: true` projects shown on the home page.
- Delete this `HANDOFF.md`.

## Commit history on the branch (newest first)

```
1c46d31 Add Medium/external articles to the writing feed
5644c3d Refine hero and about copy
8cb1986 Set experience and education from Yuan's LinkedIn profile
faf07a2 Fill in experience details from public profile sources
d99a607 Add GitHub Pages deploy workflow and README
38df5e1 Remove old generated Hugo output
bdb7a30 Add pages: home, projects, publications, blog, photos, RSS
11ae220 Migrate all content from the old Hugo output
0bc04f3 Scaffold Astro + Tailwind site shell
```
