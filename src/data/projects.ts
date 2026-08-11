export interface Project {
  name: string;
  description: string;
  url: string;
  tags: string[];
  /** Star count is a manual snapshot — update occasionally (see README). */
  stars?: number;
  featured?: boolean;
}

export const projects: Project[] = [
  {
    name: 'finetune-esm',
    description:
      'Scalable fine-tuning of ESM-2 protein language models with distributed training (FSDP/DeepSpeed via Ray) and parameter-efficient techniques such as LoRA.',
    url: 'https://github.com/naity/finetune-esm',
    tags: ['PyTorch', 'Protein LMs', 'LoRA', 'Distributed Training'],
    stars: 34,
    featured: true,
  },
  {
    name: 'image_tabular',
    description:
      'Python library for training deep learning models that combine image and tabular data in a single network, built on fastai. Used for the SIIM-ISIC melanoma classification challenge.',
    url: 'https://github.com/naity/image_tabular',
    tags: ['Deep Learning', 'fastai', 'Computer Vision'],
    stars: 91,
    featured: true,
  },
  {
    name: 'protein-transformer',
    description:
      'A step-by-step, from-scratch transformer implementation for antibody classification — training, tuning, and evaluation included.',
    url: 'https://github.com/naity/protein-transformer',
    tags: ['Transformers', 'PyTorch', 'Antibodies'],
    featured: true,
  },
  {
    name: 'ReceptorAI',
    description:
      'AI-driven T-cell receptor (TCR) matching and antigen discovery platform for immunotherapy research.',
    url: 'https://github.com/naity/ReceptorAI',
    tags: ['Immunology', 'TCR', 'Machine Learning'],
  },
  {
    name: 'citeseq_autoencoder',
    description:
      'Autoencoder-based integration of single-cell CITE-seq multi-omics data — companion code and notebooks for the blog tutorial and video series.',
    url: 'https://github.com/naity/citeseq_autoencoder',
    tags: ['Single-cell', 'Autoencoders', 'Multi-omics'],
  },
  {
    name: 'DeepUFC2',
    description:
      'Predicting UFC fight outcomes with deep learning on web-scraped fight and fighter statistics.',
    url: 'https://github.com/naity/DeepUFC2',
    tags: ['Deep Learning', 'Web Scraping'],
    stars: 34,
  },
];
