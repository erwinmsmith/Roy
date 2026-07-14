import { toolRegistry } from '../tools/index.js';
import type { Skill, SkillContext, SkillInput, SkillManifest, SkillOutput } from './types.js';
import type { ToolResult } from '../tools/types.js';

export interface UseToolWhenNeededParams {
  needed?: boolean;
  toolName?: string;
  params?: Record<string, unknown>;
  reason?: string;
}

export class UseToolWhenNeededSkill implements Skill {
  readonly name = 'use_tool_when_needed';
  readonly description = 'Call a registered runtime tool only when the task needs external evidence or execution.';
  readonly version = '0.1.0';

  constructor(
    private readonly executeTool: (
      agentId: string,
      toolName: string,
      params: Record<string, unknown>,
      reason?: string
    ) => Promise<ToolResult> = (_agentId, toolName, params) => toolRegistry.execute(toolName, params)
  ) {}

  getManifest(): SkillManifest {
    return {
      name: this.name,
      version: this.version,
      description: this.description,
      tags: ['tool-use', 'evidence', 'runtime'],
    };
  }

  validate(input: SkillInput): { valid: boolean; errors?: string[] } {
    const params = input.params as Partial<UseToolWhenNeededParams>;
    const errors: string[] = [];

    if (params.needed !== undefined && typeof params.needed !== 'boolean') {
      errors.push('needed must be a boolean when provided');
    }
    if (params.needed !== false && (typeof params.toolName !== 'string' || params.toolName.trim().length === 0)) {
      errors.push('toolName is required when needed is not false');
    }
    if (params.params !== undefined && (typeof params.params !== 'object' || params.params === null || Array.isArray(params.params))) {
      errors.push('params must be an object when provided');
    }
    if (params.reason !== undefined && typeof params.reason !== 'string') {
      errors.push('reason must be a string when provided');
    }

    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined,
    };
  }

  async execute(input: SkillInput, context: SkillContext): Promise<SkillOutput> {
    const validation = this.validate(input);
    if (!validation.valid) {
      return {
        success: false,
        error: `Validation failed: ${validation.errors?.join(', ')}`,
      };
    }

    const params = input.params as UseToolWhenNeededParams;
    if (params.needed === false || params.toolName === 'none') {
      return {
        success: true,
        result: {
          skipped: true,
          reason: params.reason ?? 'No tool was needed for this task.',
        },
        metadata: {
          agentId: context.agentId,
          toolUsed: false,
        },
      };
    }

    const toolName = params.toolName!;
    if (!toolRegistry.has(toolName)) {
      return {
        success: false,
        error: `Tool "${toolName}" not found`,
        metadata: {
          agentId: context.agentId,
          availableTools: toolRegistry.keys(),
        },
      };
    }

    const result = await this.executeTool(context.agentId, toolName, params.params ?? {}, params.reason);
    return {
      success: result.success,
      result: result.result,
      error: result.error,
      metadata: {
        ...result.metadata,
        agentId: context.agentId,
        toolUsed: true,
        toolName,
        reason: params.reason,
      },
    };
  }
}
