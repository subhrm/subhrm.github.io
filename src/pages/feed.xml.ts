import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts');
  const sortedPosts = posts.sort(
    (a, b) => new Date(b.data.date).getTime() - new Date(a.data.date).getTime()
  );

  return rss({
    title: "Subhendu's Blog",
    description: "I plan to document things in this blog for the benefit of future me.",
    site: context.site ?? 'https://subhrm.github.io',
    items: sortedPosts.map((post) => {
      const rawPath = post.filePath || post.id;
      const baseName = rawPath.split('/').pop()?.replace(/\.md$/, '') || post.id;
      const match = baseName.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)$/);
      const year = match ? match[1] : String(post.data.date.getFullYear());
      const month = match ? match[2] : String(post.data.date.getMonth() + 1).padStart(2, '0');
      const day = match ? match[3] : String(post.data.date.getDate()).padStart(2, '0');
      const slug = match ? match[4] : baseName;
      const category = Array.isArray(post.data.categories)
        ? post.data.categories[0]
        : (post.data.categories || 'post');

      return {
        title: post.data.title,
        pubDate: post.data.date,
        description: post.data.description || post.data.title,
        link: `/${category}/${year}/${month}/${day}/${slug}/`,
      };
    }),
    customData: `<language>en-us</language>`,
  });
}
