import { describe, expect, it } from 'vitest';
import {
  isSuccessfulWorkspaceMutationCall,
  isSuccessfulWorkspaceVerificationCall,
  taskRequestsWorkspaceMutation,
} from '../src/core/tools/executionIntent.js';

describe('workspace execution intent', () => {
  it('does not treat diagnostic redirection to /dev/null as a mutation', () => {
    expect(isSuccessfulWorkspaceMutationCall({
      toolName: 'shell.exec',
      params: { command: 'pip list --format=columns 2>/dev/null | grep -i langchain' },
      success: true,
    })).toBe(false);
  });

  it('does not treat temporary output as a workspace mutation', () => {
    expect(isSuccessfulWorkspaceMutationCall({
      toolName: 'shell.exec',
      params: { command: "echo 'started' > /tmp/migration_started.txt" },
      success: true,
    })).toBe(false);
    expect(isSuccessfulWorkspaceMutationCall({
      toolName: 'shell.exec',
      params: { command: "echo 'started' > /var/tmp/migration_started.txt" },
      success: true,
    })).toBe(false);
  });

  it('recognizes actual file writes and verification commands', () => {
    expect(isSuccessfulWorkspaceMutationCall({
      toolName: 'shell.exec',
      params: { command: "printf 'ready' > artifact.txt" },
      success: true,
    })).toBe(true);
    expect(isSuccessfulWorkspaceMutationCall({
      toolName: 'fs.write',
      params: { path: 'artifact.txt', content: 'ready' },
      success: true,
    })).toBe(true);
    expect(isSuccessfulWorkspaceMutationCall({
      toolName: 'fs.replace',
      params: { path: 'artifact.txt', oldText: 'ready', newText: 'done' },
      success: true,
    })).toBe(true);
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: 'pytest -q' },
      success: true,
    })).toBe(true);
  });

  it('does not accept verification commands that mask a failing exit status', () => {
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: 'pytest -q 2>&1 || true' },
      success: true,
    })).toBe(false);
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: 'npm test; true' },
      success: true,
    })).toBe(false);
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: 'set +e; npm run check' },
      success: true,
    })).toBe(false);
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: "python -m support_rag.cli answer --question test 2>&1; echo 'EXIT:'$?" },
      success: true,
      result: { exitCode: 0, stdout: 'EXIT:1' },
    })).toBe(false);
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: 'pytest -q' },
      success: true,
      result: { exitCode: 1, stdout: '1 failed' },
    })).toBe(false);
    expect(isSuccessfulWorkspaceVerificationCall({
      toolName: 'shell.exec',
      params: { command: 'pytest -q' },
      success: true,
      result: { exitCode: 0, stdout: '12 passed' },
    })).toBe(true);
  });

  it('distinguishes mutation tasks from explicitly read-only work', () => {
    expect(taskRequestsWorkspaceMutation('Migrate the project code and run tests.')).toBe(true);
    expect(taskRequestsWorkspaceMutation('Review the project in read-only mode.')).toBe(false);
  });
});
