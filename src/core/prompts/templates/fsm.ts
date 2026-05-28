// FSM state transition prompt templates

export const FSM_DIAGNOSE_PROMPT = `You are diagnosing the current reasoning trace for bottlenecks.
Current trace:
{trace}

Analyze for:
- Uncertainty: areas where confidence is low
- Conflict: unresolved disagreements or contradictions
- Missing evidence: gaps in supporting information
- Blind spots: perspectives or considerations not addressed

Return a diagnosis with:
- identified_bottlenecks: list of bottleneck types found
- severity: low/medium/high
- recommendations: suggested next actions
`;

export const FSM_DECIDE_PROMPT = `You are deciding whether to invest more thinking budget.

Context:
- Current state: {current_state}
- Remaining budget: {budget}
- Candidate investments: {candidates}

Estimate the expected return for each candidate:
- expected_gain: improvement expected (0-1)
- expected_cost: resource cost (0-1)
- expected_risk: risk of negative impact (0-1)
- net_return: expected_gain - expected_cost - expected_risk
- decision_reason: brief explanation

Output the investment with the highest positive net_return, or "stop" if none have positive return.
`;

export const FSM_DERIVE_PROMPT = `You are deriving new candidate agents or subteams.

Parent unit: {parent_unit}
Current trace: {trace}
Budget remaining: {budget}

Use evolutionary operators:
- Mutation: modify one dimension of parent (role, goal, perspective, ToM order, etc.)
- Crossover: combine useful properties from multiple existing units
- Selection: choose candidates with highest expected return

Generate candidate specifications with:
- unit_type: "agent" or "subteam"
- role: the agent's role
- goal: the agent's objective
- tom_order: 1-3 (for ToM-aware reasoning)
- expected_utility: estimated benefit
`;

export const FSM_VERIFY_PROMPT = `You are verifying whether the reasoning is complete.

Original question: {question}
Current answer: {answer}
Trace: {trace}

Check for:
- All major uncertainties resolved?
- Conflicts addressed?
- Evidence coverage adequate?
- No remaining blind spots?

Return:
- verified: boolean
- confidence: 0-1
- remaining_issues: list of unresolved items
`;

export const FSM_STATES = {
  S_solo: { name: 'S_solo', description: 'Continue reasoning without expansion' },
  S_diagnose: { name: 'S_diagnose', description: 'Diagnose current trace for bottlenecks' },
  S_decide: { name: 'S_decide', description: 'Decide if expansion is worth the cost' },
  S_derive: { name: 'S_derive', description: 'Derive new agents or subteams' },
  S_reuse: { name: 'S_reuse', description: 'Reuse cached agents or ToM groups' },
  S_execute: { name: 'S_execute', description: 'Execute selected agents or subteams' },
  S_merge: { name: 'S_merge', description: 'Merge outputs into parent representation' },
  S_verify: { name: 'S_verify', description: 'Verify resolution of issues' },
  S_backtrack: { name: 'S_backtrack', description: 'Backtrack if verification fails' },
  S_final: { name: 'S_final', description: 'Produce final answer' },
} as const;

export type FSMStateName = keyof typeof FSM_STATES;

export const fsmDiagnoseTemplate = {
  name: 'fsm_diagnose',
  description: 'FSM diagnose state prompt',
  template: FSM_DIAGNOSE_PROMPT,
};

export const fsmDecideTemplate = {
  name: 'fsm_decide',
  description: 'FSM decide state prompt',
  template: FSM_DECIDE_PROMPT,
};

export const fsmDeriveTemplate = {
  name: 'fsm_derive',
  description: 'FSM derive state prompt',
  template: FSM_DERIVE_PROMPT,
};

export const fsmVerifyTemplate = {
  name: 'fsm_verify',
  description: 'FSM verify state prompt',
  template: FSM_VERIFY_PROMPT,
};