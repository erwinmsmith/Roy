import { stableStructuralFingerprint } from './graphs.js';
import type { EpistemicClaim, EpistemicEvidence, ExternalObservation,
  InformationRealizationNode, ResidualRequirement } from './informationRealizationTypes.js';

export const GLOBAL_EPISTEMIC_STATE_SCHEMA_VERSION = 2 as const;

export interface SemanticRelation {
  leftId: string;
  rightId: string;
  label: 'entail' | 'contradict' | 'unknown';
  probabilities: { entail: number; contradict: number; unknown: number };
  model: string;
  modelRevision: string;
  requestId: string;
  candidateSource: 'minilm_top_k';
  candidateRank: number;
}

export interface RuntimeProcessEvent {
  id: string;
  kind: 'task_instruction' | 'organization_action' | 'terminal_command' | 'terminal_result' | 'file_change'
    | 'failure' | 'usage' | 'verifier';
  at: number;
  nodeId?: string;
  command?: string;
  cwd?: string;
  timeoutMs?: number;
  exitCode?: number;
  output?: string;
  attributes?: Record<string, unknown>;
}

export interface GlobalEpistemicState {
  schemaVersion: typeof GLOBAL_EPISTEMIC_STATE_SCHEMA_VERSION;
  trajectoryId: string;
  taskId: string;
  sequence: number;
  requirements: ResidualRequirement[];
  claims: EpistemicClaim[];
  assumptions: Array<Record<string, unknown>>;
  evidence: EpistemicEvidence[];
  externalObservations: ExternalObservation[];
  semanticRelations: SemanticRelation[];
  blindSpots: string[];
  dependencies: Array<Record<string, unknown>>;
  nodes: InformationRealizationNode[];
  dagEdges: Array<Record<string, unknown>>;
  activeSubtree: string[];
  /** Bounded local event projection. The complete immutable ledger lives on the session snapshot. */
  runtimeEvents: RuntimeProcessEvent[];
  runtimeEventRange?: { start: number; endExclusive: number; total: number };
  usage: { inputTokens: number; outputTokens: number; wallTimeMs: number };
  environmentRevision: string;
  previousFingerprint?: string;
  fingerprint: string;
}

export type GlobalEpistemicStateInput = Omit<GlobalEpistemicState, 'schemaVersion' | 'fingerprint'>;

export class GlobalEpistemicStateRecorder {
  private readonly states: GlobalEpistemicState[] = [];

  append(input: GlobalEpistemicStateInput): GlobalEpistemicState {
    if (input.sequence !== this.states.length) {
      throw new Error(`Expected process state M_${this.states.length}, received M_${input.sequence}`);
    }
    const previous = this.states.at(-1);
    if (input.previousFingerprint !== previous?.fingerprint) {
      throw new Error('Global epistemic state fingerprint chain is broken');
    }
    const material = structuredClone({
      schemaVersion: GLOBAL_EPISTEMIC_STATE_SCHEMA_VERSION,
      ...input,
    });
    const state = { ...material, fingerprint: stableStructuralFingerprint(material) };
    this.states.push(state);
    return structuredClone(state);
  }

  restore(states: GlobalEpistemicState[]): void {
    this.states.length = 0;
    for (const state of states) {
      const { fingerprint, ...material } = state;
      if (stableStructuralFingerprint(material) !== fingerprint) {
        throw new Error(`Process state M_${state.sequence} fingerprint is invalid`);
      }
      const previous = this.states.at(-1);
      if (state.sequence !== this.states.length
        || state.previousFingerprint !== previous?.fingerprint) {
        throw new Error('Restored process state chain is not append-only');
      }
      this.states.push(structuredClone(state));
    }
  }

  latest(): GlobalEpistemicState | undefined {
    const state = this.states.at(-1);
    return state ? structuredClone(state) : undefined;
  }

  snapshot(): GlobalEpistemicState[] {
    return structuredClone(this.states);
  }
}
