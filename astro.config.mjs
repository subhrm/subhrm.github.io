import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function mirrorPages() {
  return {
    name: 'mirror-pages',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        const distDir = fileURLToPath(dir);

        function walk(currentDir) {
          const entries = fs.readdirSync(currentDir, { withFileTypes: true });
          for (const entry of entries) {
            const fullPath = path.join(currentDir, entry.name);
            if (entry.isDirectory()) {
              walk(fullPath);
            } else if (entry.isFile() && entry.name.endsWith('.html')) {
              // If it's not index.html and not 404.html, make a directory version: /name/index.html
              if (entry.name !== 'index.html' && entry.name !== '404.html') {
                const subDirName = entry.name.replace(/\.html$/, '');
                const targetDir = path.join(currentDir, subDirName);
                if (!fs.existsSync(targetDir)) {
                  fs.mkdirSync(targetDir, { recursive: true });
                }
                const targetFile = path.join(targetDir, 'index.html');
                fs.copyFileSync(fullPath, targetFile);
              }
            }
          }
        }

        walk(distDir);
      },
    },
  };
}

export default defineConfig({
  site: 'https://subhrm.github.io',
  base: '/',
  output: 'static',
  build: {
    format: 'file'
  },
  vite: {
    plugins: [tailwindcss()],
  },
  integrations: [
    mirrorPages(),
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-light',
      wrap: true,
    },
  },
});
