export interface ExperienceEntry {
  org: string;
  role?: string;
  location?: string;
  /** Rendered only when set. */
  period?: string;
  /** Rendered only when set. */
  summary?: string;
}

// Dates and titles from Yuan's LinkedIn profile (Aug 2026).
export const experience: ExperienceEntry[] = [
  {
    org: 'Anthropic',
    role: 'Member of Technical Staff',
    period: 'Jun 2026 – Present',
    location: 'Seattle, WA',
  },
  {
    org: 'Amazon Web Services (AWS)',
    role: 'Applied Scientist, Generative AI',
    period: 'Aug 2024 – Jun 2026',
    location: 'Seattle, WA',
    summary:
      'Built generative AI applications on Amazon Bedrock for enterprise customers: LLM workflow assistants, voice AI simulation with agentic tool-calling, multi-agent systems for pharmaceutical research reports, and RAG support assistants. Also developed a fine-tuning approach for protein language models for binding-affinity prediction (oral presentation, AMLC 2025).',
  },
  {
    org: 'Gilead Sciences',
    role: 'Applied Research Scientist',
    period: 'Jan 2022 – Jul 2024',
    location: 'Seattle, WA',
    summary:
      'Machine learning and LLMs for drug discovery research: an internal RAG chatbot serving 200+ researchers, ReceptorGPT for TCR similarity search over ESM-2 embeddings, transformer models for antibody antigen specificity, and survival-model biomarker discovery for cell-therapy response.',
  },
  {
    org: 'Fred Hutchinson Cancer Center',
    role: 'Staff Data Scientist',
    period: 'Nov 2019 – Dec 2021',
    location: 'Seattle, WA',
    summary:
      'Single-cell genomics meets machine learning: immune-cell correlates of reduced HIV risk after vaccination, multimodal image + tabular deep learning for melanoma detection, and a COVID-19 single-cell portal integrating 21 multi-omics datasets spanning 3M+ cells.',
  },
  {
    org: 'La Jolla Institute for Immunology',
    role: 'Postdoctoral Fellow → Bioinformatics Scientist',
    period: 'Jun 2016 – Nov 2019',
    location: 'La Jolla, CA',
    summary:
      'Systems immunology of human T cells in the Sette lab: discovered unique phenotypes of CD4 Temra cells, profiled the transcriptomes and TCR repertoires of dengue-specific T cells, and introduced CyTOF high-dimensional single-cell analysis to the lab. AAI Intersect Fellow bridging immunology and bioinformatics.',
  },
];

export const education: ExperienceEntry[] = [
  {
    org: 'Georgia Institute of Technology',
    role: 'M.S. in Computer Science',
    period: '2024 – 2027 (expected)',
  },
  {
    org: 'University of Alabama at Birmingham',
    role: 'Ph.D. in Microbiology and Immunology',
    period: '2011 – 2016',
  },
  {
    org: 'Tianjin University',
    role: 'B.E. in Bioengineering and Biomedical Engineering',
    period: '2007 – 2011',
  },
];
