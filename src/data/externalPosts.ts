// Articles published off-site (Medium / Towards AI). Rendered as link-out
// cards in the writing feed alongside native blog posts. Add the ones worth
// featuring here — newest first is not required; the feed sorts by date.
export interface ExternalPost {
  title: string;
  /** ISO date (YYYY-MM-DD). */
  date: string;
  url: string;
  /** Publication name shown on the badge, e.g. "Towards AI" or "Medium". */
  source: string;
  description: string;
  tags?: string[];
}

export const externalPosts: ExternalPost[] = [
  {
    title: 'Building an Agentic Resume Matcher: Python Foundations for GenAI',
    date: '2025-12-14',
    url: 'https://pub.towardsai.net/building-an-agentic-resume-matcher-python-foundations-for-genai-dc77febd34b3',
    source: 'Towards AI',
    description:
      'Learning Python foundations for GenAI development through a practical agentic application: a resume matcher.',
    tags: ['GenAI', 'AI Agents', 'Python'],
  },
  {
    title: 'Building a Tool-Using LLM Agent from Scratch',
    date: '2025-03-18',
    url: 'https://medium.com/data-science-collective/building-a-tool-using-llm-agent-from-scratch-28c409aac46b',
    source: 'Data Science Collective',
    description:
      'Creating a UFC fight recommendation agent with tool-using and reasoning capabilities, without frameworks.',
    tags: ['AI Agents', 'LLM', 'GenAI'],
  },
  {
    title: 'Building Transformer Models for Proteins From Scratch',
    date: '2024-05-07',
    url: 'https://medium.com/data-science/building-transformer-models-for-proteins-from-scratch-60884eab5cc8',
    source: 'Towards Data Science',
    description:
      'A practical guide to building and evaluating protein language models, starting from the transformer architecture itself.',
    tags: ['Transformers', 'Protein Language Models', 'Deep Learning'],
  },
  {
    title: 'A Primer to Protein Science',
    date: '2024-03-14',
    url: 'https://medium.com/@yuan_tian/a-primer-to-protein-science-1b6778ae995e',
    source: 'Medium',
    description:
      'A concise introduction to protein science as background for understanding how AI is transforming protein research.',
    tags: ['Protein', 'Biology', 'AI'],
  },
  {
    title: 'Predicting T Cell Receptor Specificity with Deep Learning',
    date: '2019-02-23',
    url: 'https://becominghuman.ai/predicting-t-cell-receptor-specificity-with-deep-learning-12757a899e8b',
    source: 'Becoming Human',
    description:
      'Predicting which epitope a T cell receptor recognizes with embeddings and convolutional neural networks on VDJdb data.',
    tags: ['Immunology', 'Deep Learning', 'TCR'],
  },
];
