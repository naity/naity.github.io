// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://naity.github.io',
  integrations: [sitemap()],
  vite: {
    plugins: [tailwindcss()],
  },
  // Old Hugo URLs → new locations (meta-refresh stubs in the static build).
  redirects: {
    // Blog posts used to live at root-level slugs.
    '/ai-takes-on-the-ufc-predict-ufc-fight-outcomes-with-chatgpt':
      '/blog/ai-takes-on-the-ufc-predict-ufc-fight-outcomes-with-chatgpt/',
    '/integrating-image-and-tabular-data-for-deep-learning':
      '/blog/integrating-image-and-tabular-data-for-deep-learning/',
    '/integrative-analysis-of-single-cell-multi-omics-data-using-deep-learning':
      '/blog/integrative-analysis-of-single-cell-multi-omics-data-using-deep-learning/',
    '/refocus-making-out-of-focus-microscopy-images-in-focus-again':
      '/blog/refocus-making-out-of-focus-microscopy-images-in-focus-again/',
    '/posts': '/blog/',
    // Old per-publication pages → anchored entries on the single publications page.
    ...Object.fromEntries(
      [
        'cd4-temra',
        'covid-review',
        'cytof',
        'cytotoxic-review',
        'dengue-antigens',
        'dengue-review',
        'dengue_cd4',
        'dengue_cd8',
        'epitope-review',
        'flavivirus-review',
        'hiv',
        'il-10-paper',
        'il-21-paper',
        'il21-review',
        'leakage',
        'normalization',
        'pertussis',
        'superscan',
        'zika',
      ].map((slug) => [`/publications/${slug}`, `/publications/#${slug}`])
    ),
  },
});
