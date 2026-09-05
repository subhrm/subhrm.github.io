import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';

// remark-math only emits math-display for fenced $$ blocks; promote
// paragraphs that consist solely of inline math (e.g. a single-line
// $$...$$) to display math so MathJax renders them as blocks.
function promoteDisplayMath() {
  function isInlineMathCode(node) {
    return (
      node &&
      node.type === 'element' &&
      node.tagName === 'code' &&
      Array.isArray(node.properties?.className) &&
      node.properties.className.includes('math-inline')
    );
  }

  function visit(node) {
    if (!node.children) return;
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      const isWhitespace = child.type === 'text' && /^\s*$/.test(child.value);
      if (
        node.tagName === 'p' &&
        child.type === 'element' &&
        isInlineMathCode(child)
      ) {
        // Promote when every non-whitespace child of the paragraph is math.
        const allMath = node.children.every(
          (c) => isInlineMathCode(c) || (c.type === 'text' && /^\s*$/.test(c.value))
        );
        if (allMath && node.children.some(isInlineMathCode)) {
          const blocks = [];
          for (const c of node.children) {
            if (isInlineMathCode(c)) {
              c.properties.className = ['language-math', 'math-display'];
              blocks.push({
                type: 'element',
                tagName: 'pre',
                properties: {},
                children: [c],
              });
            }
          }
          node.children.splice(i, node.children.length, ...blocks);
          for (const b of blocks) visit(b);
          continue;
        }
      }
      visit(child);
    }
  }

  return (tree) => visit(tree);
}

export default defineConfig({
  site: 'https://subhrm.github.io',
  base: '/',
  output: 'static',
  build: {
    format: 'directory',
  },
  trailingSlash: 'always',
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [promoteDisplayMath],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      wrap: false,
    },
  },
});
