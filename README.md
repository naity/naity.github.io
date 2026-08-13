# naity.github.io

Personal site of Yuan Tian — built with [Astro](https://astro.build) + [Tailwind CSS](https://tailwindcss.com), deployed to GitHub Pages at [naity.github.io](https://naity.github.io).

## Development

```sh
npm install
npm run dev        # local dev server at localhost:4321
npm run build      # production build into dist/
npm run preview    # serve the production build locally
npm run check      # type-check .astro files
```

## Where things live

| Content            | Location                                       |
| ------------------ | ---------------------------------------------- |
| Bio, socials, nav  | `src/data/site.ts`                             |
| Projects           | `src/data/projects.ts`                         |
| Experience         | `src/data/experience.ts`                       |
| Photos (gallery)   | `src/data/photos.ts` + `src/assets/photos/`    |
| Blog posts         | `src/content/blog/<slug>/index.md`             |
| Publications       | `src/content/publications/<slug>/index.md`     |
| Publication PDFs   | `public/publications/<slug>/` (stable URLs)    |
| Design tokens      | `src/styles/global.css`                        |

> **Note:** star counts in `src/data/projects.ts` are a manual snapshot; refresh
> occasionally. In `src/data/experience.ts`, update the Georgia Tech entry once
> the degree is completed.

## Adding a blog post

Create `src/content/blog/my-post-slug/index.md`:

```md
---
title: "Post title"
date: 2026-01-15
description: "One-sentence summary used in cards, SEO, and RSS."
tags: ["Deep Learning"]
cover: ./featured-image.png # optional, colocated in the same folder
---

Post body in Markdown. Colocated images work with relative paths.
```

## Adding a publication

Create `src/content/publications/my-paper/index.md` (the body is the abstract):

```md
---
title: "Paper title"
authors: ["First Author", "Yuan Tian", "Last Author"] # "Yuan Tian" gets bolded
journal: "Journal Name 12(3), 456-789"
date: 2026-01-15
type: paper # or: preprint
doi: "10.1234/example" # optional
pdf: "/publications/my-paper/paper.pdf" # optional; put the file in public/publications/my-paper/
image: "./featured-image.jpg" # optional thumbnail
---

Abstract text here.
```

## Adding photos

1. Run `node scripts/resize-photos.mjs <folder-or-file-with-originals>` — writes
   web-sized JPEGs into `src/assets/photos/`.
2. Add an entry (import + alt text) to `src/data/photos.ts`.

## Deployment

Pushes to `main` build and deploy automatically via `.github/workflows/deploy.yml`.

**One-time setup after merging the redesign:** in the repo settings, set
**Settings → Pages → Build and deployment → Source** to **GitHub Actions**
(the site previously deployed straight from the branch). Then, optionally,
resubmit `https://naity.github.io/sitemap-index.xml` in Google Search Console.

### Old URLs

The Hugo-era URLs redirect: root-level post slugs → `/blog/<slug>/`, and
`/publications/<slug>` → anchored entries on `/publications/`. Publication PDF
URLs are unchanged. The RSS feed remains at `/index.xml`.
