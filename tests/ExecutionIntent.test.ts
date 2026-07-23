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
  });

  it('distinguishes mutation tasks from explicitly read-only work', () => {
    expect(taskRequestsWorkspaceMutation('Migrate the project code and run tests.')).toBe(true);
    expect(taskRequestsWorkspaceMutation('Review the project in read-only mode.')).toBe(false);
  });
});
