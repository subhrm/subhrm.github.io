# subhrm.github.io

Personal technical blog built with [Astro 7](https://astro.build/) and [Tailwind CSS v4](https://tailwindcss.com/), hosted statically on [GitHub Pages](https://pages.github.com/).

---

## Local Development

### Prerequisites
- [Node.js](https://nodejs.org/) (v24+ recommended)
- [npm](https://www.npmjs.com/)

### Local setup 

```bash
# Install dependencies
npm install

# Start local development server (http://localhost:4321)
npm run dev

# Build production static bundle to dist/
npm run build

# Preview production build locally
npm run preview
```

---

## Content Authoring

### Adding a New Post
1. Create a new Markdown file in `src/content/posts/` following the naming convention:
   ```
   src/content/posts/YYYY-MM-DD-post-title-slug.md
   ```
2. Include the required frontmatter:
   ```yaml
   ---
   title: "Your Post Title Here"
   date: yyyy-mm-dd hh:mm:ss +HHMM
   categories: categories
   description: "A brief summary for SEO and RSS feeds."
   tags: ["tag1", "tag2"]
   ---
   ```
