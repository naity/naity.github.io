export interface ExperienceEntry {
  org: string;
  /** Rendered only when set — the most recent entry stays deliberately title-free. */
  role?: string;
  location?: string;
  /** Rendered only when set. */
  period?: string;
  /** Rendered only when set. */
  summary?: string;
}

// Verified against public sources (LinkedIn snippets, ResearchGate, Journal of
// Cellular Immunology editor bio). Remaining TODO(yuan) items are dates/names
// only visible on the full LinkedIn profile.
export const experience: ExperienceEntry[] = [
  {
    org: 'Amazon Web Services (AWS)',
    period: '2023 — Present', // TODO(yuan): confirm start year
    location: 'Seattle, WA',
    summary:
      'Building generative AI and machine learning applications that solve real-world problems.',
  },
  {
    org: 'Fred Hutchinson Cancer Center',
    role: 'Staff Scientist',
    period: '2020 — 2023', // TODO(yuan): confirm years
    location: 'Seattle, WA',
    summary:
      'Computational immunology in the Vaccine and Infectious Disease Division and the Translational Data Science IRC — single-cell multi-omics of immune responses in vaccine trials and cancer immunotherapy, including CITE-seq normalization (ADTnorm) and T-cell profiling in multiple myeloma.',
  },
  {
    org: 'La Jolla Institute for Immunology',
    role: 'Postdoctoral Fellow',
    period: '2016 — 2020', // TODO(yuan): confirm end year
    location: 'La Jolla, CA',
    summary:
      'Computational immunology in the Sette lab — transcriptomic and TCR profiling of virus-specific T cells (dengue, Zika, pertussis). AAI Intersect Fellow bridging immunology and bioinformatics with the Peters lab.',
  },
  {
    org: 'Georgia Institute of Technology',
    role: 'M.S. in Computer Science', // TODO(yuan): confirm program name (e.g. OMSCS)
    // TODO(yuan): add years, e.g. period: '2021 — 2023'
  },
  {
    org: 'University of Alabama at Birmingham',
    role: 'Ph.D. in Microbiology and Immunology',
    period: '2011 — 2016',
    location: 'Birmingham, AL',
    summary:
      'Doctoral research on cytokine regulation of T cell differentiation and memory (IL-21, IL-10) during viral infection, in the Zajac lab.',
  },
  {
    org: 'Tianjin University',
    role: 'B.E. in Bioengineering',
    period: '2011',
    location: 'Tianjin, China',
  },
];
