# AGENTS.md

This document provides context, architectural guidelines, and development workflows for AI coding agents and developers working on this repository.

---

## 1. Project Overview

- **Repository**: `subhrm.github.io` (GitHub Pages personal blog)
- **Framework**: [Astro 7](https://astro.build/) (Static Site Generation)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) with `@tailwindcss/vite`
- **Language**: TypeScript (`strict` mode)
- **Deployment**: Automated GitHub Pages via GitHub Actions (`.github/workflows/deploy.yml`)

---

## 2. Directory Structure

```
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Pages build & deploy action
├── public/
│   └── favicon.ico                 # Static assets copied directly to root
├── src/
│   ├── components/
│   │   ├── Header.astro            # Global top navigation bar
│   │   └── Footer.astro            # Global footer with social & RSS links
│   ├── content/
│   │   └── posts/                  # Markdown blog posts (YYYY-MM-DD-slug.md)
│   ├── layouts/
│   │   ├── BaseLayout.astro        # HTML shell, SEO meta, MathJax, Mermaid
│   │   └── PostLayout.astro        # Article layout with date & category tags
│   ├── pages/
│   │   ├── [category]/[year]/[month]/[day]/
│   │   │   └── [slug].astro        # Dynamic route for posts (Jekyll permalinks)
│   │   ├── about/
│   │   │   └── index.astro         # /about/ page
│   │   ├── 404.astro               # Custom 404 error page
│   │   ├── feed.xml.ts             # Dynamic RSS 2.0 feed generator
│   │   └── index.astro             # Home page with article listings
│   ├── styles/
│   │   └── global.css              # Tailwind v4 import, typography & prose
│   └── content.config.ts           # Astro 5/7 content collections & Zod schema
├── astro.config.mjs                # Astro configuration & route mirror hook
├── package.json                    # Project dependencies and scripts
└── tsconfig.json                   # TypeScript configuration
```

---

## 3. Content Collections & Adding New Posts

### Post Location & File Naming
Store all blog posts in `src/content/posts/` with the filename pattern:
`YYYY-MM-DD-Post-Title-Slug.md`

### Frontmatter Schema (`src/content.config.ts`)
Each post must adhere to the following schema:

```yaml
---
title: "Post Title Here"
date: 2026-05-22 01:05:00 +0530 # or ISO-8601 string / Date
categories: llm                  # string or array of strings (e.g. ['llm', 'ai'])
description: "Brief summary"    # Optional description for SEO and RSS
layout: post                    # Optional legacy compatibility field
tags: []                        # Optional array of tags
---
```

---

## 4. Key Architectural Features

### Backward-Compatible Permalinks
To avoid breaking external links from the previous Jekyll setup:
- Routes generate `/:category/:year/:month/:day/:slug.html` (e.g. `/llm/2026/05/22/Mixture-of-Experts.html`).
- The `mirrorPages` integration in `astro.config.mjs` also generates directory index files (e.g. `/llm/2026/05/22/Mixture-of-Experts/index.html`) so both `.html` and trailing slash URLs work.

### LaTeX Math Support
- Math formulas are handled via MathJax 3 in `BaseLayout.astro`.
- Inline math: `$formula$` or `\(formula\)`.
- Block/display math: `$$formula$$` or `\[formula\]`.

### Mermaid Diagram Support
- Standard Mermaid code fences (```` ```mermaid ````) are dynamically initialized by client-side Mermaid in `BaseLayout.astro`.

### RSS Feed
- Dynamic RSS feed is available at `/feed.xml` generated via `@astrojs/rss` in `src/pages/feed.xml.ts`.

---

## 5. Development & Build Commands

```bash
# Install dependencies
npm install

# Start local dev server (default port: 4321)
npm run dev

# Build production bundle to dist/
npm run build

# Preview production build locally
npm run preview
```

---

## 6. Guidelines for Agents

1. **Do not remove MathJax or Mermaid scripts** from `BaseLayout.astro` as long-form technical reports rely on them.
2. **Preserve post URL structure** (`/:category/:year/:month/:day/:slug.html`) when creating or modifying post routing.
3. **Always run `npm run build`** after making structural or schema changes to ensure static compilation succeeds with 0 errors.
