export interface ExperienceEntry {
  org: string;
  /** Rendered only when set — the most recent entry stays deliberately title-free. */
  role?: string;
  location?: string;
  period: string;
  summary: string;
}

// TODO(yuan): verify orgs, roles, and years below — periods are estimates
// reconstructed from the public record (publication affiliations); adjust freely.
export const experience: ExperienceEntry[] = [
  {
    org: 'Industry — AI/ML', // TODO(yuan): add company name here if desired
    period: '2023 — Present', // TODO(yuan): confirm start year
    location: 'Seattle, WA',
    summary:
      'Building generative AI and machine learning applications that solve real-world problems.',
  },
  {
    org: 'Fred Hutchinson Cancer Center',
    role: 'Computational Biology', // TODO(yuan): exact title (Vaccine and Infectious Disease Division)
    period: '2020 — 2023', // TODO(yuan): confirm years
    location: 'Seattle, WA',
    summary:
      'Single-cell multi-omics of immune responses in vaccine trials and cancer immunotherapy — including CITE-seq normalization methods (ADTnorm) and T-cell profiling in multiple myeloma.',
  },
  {
    org: 'La Jolla Institute for Immunology',
    role: 'Postdoctoral Fellow',
    period: '2016 — 2020', // TODO(yuan): confirm years
    location: 'La Jolla, CA',
    summary:
      'Computational immunology in the Sette lab — transcriptomic and TCR profiling of virus-specific T cells (dengue, Zika, pertussis). AAI Intersect Fellow bridging immunology and bioinformatics with the Peters lab.',
  },
  {
    org: 'University of Alabama at Birmingham', // TODO(yuan): confirm institution
    role: 'Ph.D. in Immunology', // TODO(yuan): confirm degree/program
    period: '2010 — 2016', // TODO(yuan): confirm years
    location: 'Birmingham, AL',
    summary:
      'Doctoral research on cytokine regulation of T cell differentiation and memory (IL-21, IL-10) during viral infection, in the Zajac lab.',
  },
];
