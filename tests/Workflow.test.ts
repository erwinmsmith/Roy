import { describe, expect, it } from 'vitest';
import { AsyncioExecutor } from '../src/core/executor/Executor.js';
import { SignalBus } from '../src/core/executor/SignalBus.js';
import { Workflow } from '../src/core/workflow/Workflow.js';
import type { WorkflowResult } from '../src/core/workflow/WorkflowState.js';

class InputWorkflow extends Workflow<string> {
  async run(): Promise<WorkflowResult<string>> {
    this.updateState({ status: 'running' });
    const value = await this.waitForInput('Choose a deployment target');
    this.updateState({ status: 'completed' });
    return { success: true, value, metadata: this.getMetadata() };
  }
}

describe('Workflow lifecycle', () => {
  it('exposes waiting context and removes it after receiving a signal', async () => {
    const signalBus = new SignalBus();
    const executor = new AsyncioExecutor({}, signalBus);
    const workflow = new InputWorkflow(executor, { name: 'deploy', metadata: { owner: 'Roy' } });
    const run = workflow.run();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(workflow.getState()).toMatchObject({
      status: 'waiting',
      metadata: { owner: 'Roy', waitingFor: 'Choose a deployment target' },
    });
    await executor.signal('human_input:deploy', 'staging');

    await expect(run).resolves.toMatchObject({ success: true, value: 'staging' });
    expect(workflow.getState().status).toBe('completed');
    expect(workflow.getMetadata()).toEqual({ owner: 'Roy' });
    await executor.cleanup();
    signalBus.cleanup();
  });

  it('returns defensive state and metadata snapshots', () => {
    const workflow = new InputWorkflow(new AsyncioExecutor(), { metadata: { stable: true } });
    const snapshot = workflow.getState();
    snapshot.metadata.stable = false;
    const metadata = workflow.getMetadata();
    metadata.extra = 'mutated';

    expect(workflow.getMetadata()).toEqual({ stable: true });
  });
});
