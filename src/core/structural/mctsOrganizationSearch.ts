export interface MCTSCandidateLike {
  id: string;
  kind: string;
}

export interface MCTSExpandedChild<State, Candidate extends MCTSCandidateLike> {
  candidate: Candidate;
  state: State;
  prior: number;
  targetValue: number;
  terminal: boolean;
}

export interface MCTSExpansion<State, Candidate extends MCTSCandidateLike> {
  targetValue: number;
  targetRevision: number;
  actorPriors: Record<string, number>;
  children: Array<MCTSExpandedChild<State, Candidate>>;
}

interface SearchNode<State, Candidate extends MCTSCandidateLike> {
  state: State;
  remaining: Candidate[];
  depth: number;
  terminal: boolean;
  targetValue?: number;
  targetRevision?: number;
  actorPriors?: Record<string, number>;
  visits: number;
  edges?: Array<SearchEdge<State, Candidate>>;
}

interface SearchEdge<State, Candidate extends MCTSCandidateLike> {
  candidate: Candidate;
  prior: number;
  visits: number;
  valueSum: number;
  child: SearchNode<State, Candidate>;
}

export interface MCTSSearchResult<Candidate extends MCTSCandidateLike> {
  candidate: Candidate;
  rootTargetValue: number;
  selectedChildTargetValue: number;
  selectedProcessReward: number;
  targetRevision: number;
  actorPriors: Record<string, number>;
  visitCounts: Record<string, number>;
  behaviorProbabilities: Record<string, number>;
  trace: Array<Record<string, unknown>>;
}

export async function searchOrganizationMCTS<State, Candidate extends MCTSCandidateLike>(options: {
  rootState: State;
  candidates: Candidate[];
  expand: (state: State, candidates: Candidate[]) => Promise<MCTSExpansion<State, Candidate>>;
  simulations: number;
  maximumDepth: number;
  cpuCT: number;
  temperature: number;
  seed: number;
}): Promise<MCTSSearchResult<Candidate>> {
  if (options.simulations < 1 || options.maximumDepth < 1 || options.cpuCT < 0
    || options.temperature <= 0) throw new Error('Invalid MCTS configuration');
  const root: SearchNode<State, Candidate> = { state: options.rootState,
    remaining: options.candidates, depth: 0, terminal: false, visits: 0 };
  const trace: Array<Record<string, unknown>> = [];
  await ensureExpanded(root, options.expand, trace);
  if (!root.edges?.length || root.targetValue === undefined) {
    throw new Error('MCTS root has no valid organization child');
  }
  for (let simulation = 0; simulation < options.simulations; simulation += 1) {
    const leafValue = await simulate(root, options, trace, simulation);
    trace.push({ phase: 'simulation_complete', simulation, leafTargetValue: leafValue });
  }
  const visitCounts = Object.fromEntries(root.edges.map(edge => [edge.candidate.id, edge.visits]));
  const weights = root.edges.map(edge => Math.pow(edge.visits, 1 / options.temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  const probabilities = total > 0
    ? weights.map(value => value / total)
    : root.edges.map(edge => edge.prior);
  const normalizedTotal = probabilities.reduce((sum, value) => sum + value, 0);
  const behavior = probabilities.map(value => value / normalizedTotal);
  const selectedIndex = sampleIndex(behavior, options.seed);
  const selected = root.edges[selectedIndex];
  return {
    candidate: selected.candidate,
    rootTargetValue: root.targetValue,
    selectedChildTargetValue: selected.child.targetValue ?? root.targetValue,
    selectedProcessReward: (selected.child.targetValue ?? root.targetValue) - root.targetValue,
    targetRevision: root.targetRevision ?? 0,
    actorPriors: root.actorPriors ?? {}, visitCounts,
    behaviorProbabilities: Object.fromEntries(
      root.edges.map((edge, index) => [edge.candidate.id, behavior[index]])
    ), trace,
  };
}

async function ensureExpanded<State, Candidate extends MCTSCandidateLike>(
  node: SearchNode<State, Candidate>,
  expand: (state: State, candidates: Candidate[]) => Promise<MCTSExpansion<State, Candidate>>,
  trace: Array<Record<string, unknown>>
): Promise<void> {
  if (node.edges || node.terminal) return;
  const expansion = await expand(node.state, node.remaining);
  node.targetValue = expansion.targetValue;
  node.targetRevision = expansion.targetRevision;
  node.actorPriors = expansion.actorPriors;
  const priorTotal = expansion.children.reduce((sum, child) => sum + Math.max(0, child.prior), 0);
  if (expansion.children.length && priorTotal <= 0) {
    throw new Error('MCTS expansion has no positive actor prior');
  }
  node.edges = expansion.children.map(child => ({ candidate: child.candidate,
    prior: Math.max(0, child.prior) / priorTotal, visits: 0, valueSum: 0,
    child: { state: child.state,
      remaining: node.remaining.filter(value => value.id !== child.candidate.id),
      depth: node.depth + 1, terminal: child.terminal,
      targetValue: child.targetValue, targetRevision: expansion.targetRevision, visits: 0 } }));
  trace.push({ phase: 'expansion', depth: node.depth, targetValue: node.targetValue,
    targetRevision: node.targetRevision,
    candidates: node.edges.map(edge => ({ candidateId: edge.candidate.id,
      prior: edge.prior, childTargetValue: edge.child.targetValue,
      immediateProcessReward: (edge.child.targetValue ?? node.targetValue ?? 0)
        - (node.targetValue ?? 0), terminal: edge.child.terminal })) });
}

async function simulate<State, Candidate extends MCTSCandidateLike>(
  node: SearchNode<State, Candidate>,
  options: {
    expand: (state: State, candidates: Candidate[]) => Promise<MCTSExpansion<State, Candidate>>;
    maximumDepth: number; cpuCT: number;
  }, trace: Array<Record<string, unknown>>, simulation: number
): Promise<number> {
  if (node.terminal || node.depth >= options.maximumDepth || node.remaining.length === 0) {
    node.visits += 1;
    return node.targetValue ?? 0;
  }
  await ensureExpanded(node, options.expand, trace);
  if (!node.edges?.length) {
    node.visits += 1;
    return node.targetValue ?? 0;
  }
  const rootVisits = Math.max(1, node.visits);
  const edge = [...node.edges].sort((left, right) => {
    const leftScore = (left.visits ? left.valueSum / left.visits : 0)
      + options.cpuCT * left.prior * Math.sqrt(rootVisits) / (1 + left.visits);
    const rightScore = (right.visits ? right.valueSum / right.visits : 0)
      + options.cpuCT * right.prior * Math.sqrt(rootVisits) / (1 + right.visits);
    return rightScore - leftScore || left.candidate.id.localeCompare(right.candidate.id);
  })[0];
  trace.push({ phase: 'selection', simulation, depth: node.depth,
    candidateId: edge.candidate.id, visitsBefore: edge.visits,
    q: edge.visits ? edge.valueSum / edge.visits : 0, prior: edge.prior });
  const leafValue = await simulate(edge.child, options, trace, simulation);
  const backedUp = leafValue - (node.targetValue ?? 0);
  edge.visits += 1;
  edge.valueSum += backedUp;
  node.visits += 1;
  trace.push({ phase: 'backup', simulation, depth: node.depth,
    candidateId: edge.candidate.id, backedUpValue: backedUp,
    visitsAfter: edge.visits, qAfter: edge.valueSum / edge.visits });
  return leafValue;
}

function sampleIndex(probabilities: number[], seed: number): number {
  let state = Math.trunc(seed) >>> 0;
  state += 0x6d2b79f5;
  let value = state;
  value = Math.imul(value ^ value >>> 15, value | 1);
  value ^= value + Math.imul(value ^ value >>> 7, value | 61);
  const draw = ((value ^ value >>> 14) >>> 0) / 4294967296;
  let cumulative = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index];
    if (draw < cumulative) return index;
  }
  return probabilities.length - 1;
}
