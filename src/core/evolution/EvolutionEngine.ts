export interface EvolutionEvaluation<TCandidate> {
  candidate: TCandidate;
  score: number;
  breakdown: Record<string, number>;
}

export interface EvolutionRun<TSeed, TCandidate> {
  proposed: TCandidate[];
  evaluated: Array<EvolutionEvaluation<TCandidate>>;
  selected?: EvolutionEvaluation<TCandidate>;
  seed: TSeed;
}

export interface EvolutionStrategy<TSeed, TCandidate> {
  propose(seed: TSeed): Promise<TCandidate[]> | TCandidate[];
  evaluate(candidates: TCandidate[], seed: TSeed): Promise<Array<EvolutionEvaluation<TCandidate>>>;
  select(evaluated: Array<EvolutionEvaluation<TCandidate>>, seed: TSeed): EvolutionEvaluation<TCandidate> | undefined;
}

export class EvolutionEngine<TSeed, TCandidate> {
  constructor(private readonly strategy: EvolutionStrategy<TSeed, TCandidate>) {}

  async run(seed: TSeed): Promise<EvolutionRun<TSeed, TCandidate>> {
    const proposed = await this.strategy.propose(seed);
    const evaluated = await this.strategy.evaluate(proposed, seed);
    const selected = this.strategy.select(evaluated, seed);
    return { seed, proposed, evaluated, selected };
  }
}
