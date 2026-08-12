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
    date: '2025-12-14', // TODO(yuan): confirm exact publication date
    url: 'https://pub.towardsai.net/building-an-agentic-resume-matcher-python-foundations-for-genai-dc77febd34b3',
    source: 'Towards AI',
    description:
      'Python foundations for building an agentic resume matcher with generative AI, the starting point for a hands-on GenAI series.',
    tags: ['GenAI', 'AI Agents', 'Python'],
  },
];
