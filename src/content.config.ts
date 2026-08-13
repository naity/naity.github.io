import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Entries live at <collection>/<slug>/index.md with images colocated;
// the custom generateId keeps entry ids equal to the folder slug.
const stripIndex = ({ entry }: { entry: string }) => entry.replace(/\/index\.md$/, '');

const blog = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './src/content/blog', generateId: stripIndex }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      date: z.coerce.date(),
      description: z.string(),
      tags: z.array(z.string()).default([]),
      cover: image().optional(),
    }),
});

const publications = defineCollection({
  loader: glob({
    pattern: '**/index.md',
    base: './src/content/publications',
    generateId: stripIndex,
  }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      authors: z.array(z.string()),
      journal: z.string(),
      date: z.coerce.date(),
      type: z.enum(['paper', 'preprint']),
      /** Bare DOI, e.g. "10.1038/s41467-025-61023-6" — rendered as https://doi.org/<doi>. */
      doi: z.string().optional(),
      /** Site-absolute path to a locally hosted PDF, e.g. "/publications/<slug>/<file>.pdf". */
      pdf: z.string().optional(),
      image: image().optional(),
    }),
});

export const collections = { blog, publications };
