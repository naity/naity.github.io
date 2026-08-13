export const site = {
  name: 'Yuan Tian',
  tagline: 'Building AI/ML applications to solve real-world problems.',
  description:
    'Yuan Tian builds AI/ML applications to solve real-world problems, bringing a background in computational immunology research to generative AI and machine learning.',
  /** Keep in sync with `site` in astro.config.mjs, which drives canonical URLs. */
  url: 'https://naity.github.io',
  location: 'Seattle, WA',
  email: 'ytiancompbio@gmail.com',
} as const;

export const nav = [
  { label: 'Projects', href: '/projects/' },
  { label: 'Publications', href: '/publications/' },
  { label: 'Blog', href: '/blog/' },
  { label: 'Photos', href: '/photos/' },
] as const;

export type SocialIconName =
  | 'github'
  | 'linkedin'
  | 'x'
  | 'scholar'
  | 'medium'
  | 'youtube'
  | 'email'
  | 'rss';

export interface SocialLink {
  label: string;
  href: string;
  icon: SocialIconName;
}

export const socials: SocialLink[] = [
  { label: 'GitHub', href: 'https://github.com/naity', icon: 'github' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/ytian-aiml/', icon: 'linkedin' },
  { label: 'X (Twitter)', href: 'https://x.com/ytiancompbio', icon: 'x' },
  {
    label: 'Google Scholar',
    href: 'https://scholar.google.com/citations?user=8s-gqV0AAAAJ',
    icon: 'scholar',
  },
  { label: 'Medium', href: 'https://medium.com/@yuan_tian', icon: 'medium' },
  { label: 'YouTube', href: 'https://www.youtube.com/@ytiancompbio', icon: 'youtube' },
  { label: 'Email', href: 'mailto:ytiancompbio@gmail.com', icon: 'email' },
  { label: 'RSS', href: '/index.xml', icon: 'rss' },
];
