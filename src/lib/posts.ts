import { getCollection, type CollectionEntry } from 'astro:content';
import { externalPosts, type ExternalPost } from '../data/externalPosts';

export type FeedEntry =
  | { kind: 'native'; date: Date; native: CollectionEntry<'blog'> }
  | { kind: 'external'; date: Date; external: ExternalPost };

/** Native blog posts and external (Medium) articles, merged and sorted newest first. */
export async function getWritingFeed(): Promise<FeedEntry[]> {
  const native = await getCollection('blog');
  const entries: FeedEntry[] = [
    ...native.map((post) => ({ kind: 'native' as const, date: post.data.date, native: post })),
    ...externalPosts.map((post) => ({
      kind: 'external' as const,
      date: new Date(post.date),
      external: post,
    })),
  ];
  return entries.sort((a, b) => b.date.valueOf() - a.date.valueOf());
}
