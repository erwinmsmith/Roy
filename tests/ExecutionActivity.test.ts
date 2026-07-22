import { describe, expect, it } from 'vitest';
import { RootExecutionActivityProjector } from '../src/core/runtime/executionActivity.js';
import type {
  RootExecutionActivity,
  RootExecutionStep,
  RootExecutionTreeState,
} from '../src/core/runtime/executionTree.js';

describe('Root execution activity projection', () => {
  it('checkpoints only successful tool results as evidence', () => {
    const projector = new RootExecutionActivityProjector();
    const step = {
      id: 'correlation.step_01',
      index: 1,
      decision: { action: 'delegate', reason: 'Inspect evidence.', agentCount: 1 },
      actorIds: ['agent_researcher_001'],
      teamIds: [],
    } as RootExecutionStep;
    const tree = {
      correlationId: 'correlation',
      task: 'Inspect evidence.',
      rootAgentId: 'root',
    } as RootExecutionTreeState;
    const activities: RootExecutionActivity[] = [
      activity('approval', 'tool.approval.requested', { toolName: 'shell.exec' }),
      activity('call', undefined, { kind: 'tool.call' }),
      activity('failed-result', 'tool.result', { toolName: 'fs.read', success: false }),
      activity('result', 'tool.result', { toolName: 'fs.list', success: true }, 'Observed src/ and tests/.'),
    ];

    const checkpoint = projector.checkpoint({ tree, step, activities });

    expect(checkpoint.evidence).toEqual(['Observed src/ and tests/.']);
  });
});

function activity(
  id: string,
  eventType: string | undefined,
  data: Record<string, unknown>,
  summary = id
): RootExecutionActivity {
  return {
    id,
    kind: 'tool',
    status: 'completed',
    label: id,
    eventType,
    summary,
    startedAt: 1,
    completedAt: 1,
    data,
  };
}
