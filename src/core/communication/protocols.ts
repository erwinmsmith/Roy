import type {
  AgentCommunicationContext,
  AgentCommunicationProtocol,
  CommunicationProtocolInput,
  MultiPartyTrace,
} from './types.js';

function payloadText(payload: unknown): string {
  if (typeof payload === 'string') return payload;
  try {
    return JSON.stringify(payload, null, 2);
  } catch {
    return String(payload);
  }
}

function renderTrace(trace: MultiPartyTrace): string {
  const recipients = trace.to.map(actor => actor.id).join(', ') || 'none';
  const content = trace.content ? ` content=${JSON.stringify(trace.content)}` : '';
  return `- [${trace.phase}] ${trace.from.id} -> ${recipients} ${trace.kind}${content}`;
}

export class StructuredCommunicationProtocol implements AgentCommunicationProtocol {
  readonly id = 'structured';
  readonly version = '1.0';
  readonly description = 'Simple provider-neutral structured message and multi-party trace template.';

  render(input: CommunicationProtocolInput): AgentCommunicationContext {
    const message = input.message;
    const rendered = [
      `<agent_communication protocol="${this.id}" version="${this.version}">`,
      '<message>',
      `id: ${message.id}`,
      `kind: ${message.kind}`,
      `from: ${message.from}`,
      `to: ${message.to}`,
      `correlation: ${message.correlationId ?? 'none'}`,
      input.task ? `task: ${input.task}` : undefined,
      'payload:',
      payloadText(message.payload),
      '</message>',
      '<multi_party_trace>',
      input.traces.length > 0 ? input.traces.map(renderTrace).join('\n') : '- none',
      '</multi_party_trace>',
      '</agent_communication>',
    ].filter(value => value !== undefined).join('\n');

    return {
      protocolId: this.id,
      protocolVersion: this.version,
      messageId: message.id,
      correlationId: message.correlationId,
      rendered,
      traces: input.traces,
      metadata: { participantIds: input.participants.map(item => item.actor.id) },
    };
  }
}

export class ToMCommunicationProtocol implements AgentCommunicationProtocol {
  readonly id = 'tom';
  readonly version = '1.0';
  readonly description = 'Theory-of-Mind communication template for beliefs, goals, uncertainty, perspectives, and observable traces.';

  private readonly structured = new StructuredCommunicationProtocol();

  render(input: CommunicationProtocolInput): AgentCommunicationContext {
    const base = this.structured.render(input);
    const profile = input.recipient.tomProfile;
    const participantModels = input.participants.map(participant => ({
      id: participant.actor.id,
      type: participant.actor.type,
      parentId: participant.actor.parentId,
      teamId: participant.actor.teamId,
      perspective: participant.tomProfile?.perspective,
      goals: participant.tomProfile?.goalModel ?? [],
      uncertainty: participant.tomProfile?.uncertainty ?? [],
    }));
    const rendered = [
      `<agent_communication protocol="${this.id}" version="${this.version}">`,
      '<recipient_model>',
      `actor: ${input.recipient.actor.id}`,
      `perspective: ${profile?.perspective ?? 'unspecified'}`,
      `belief_scope: ${JSON.stringify(profile?.beliefScope ?? [])}`,
      `goals: ${JSON.stringify(profile?.goalModel ?? [])}`,
      `uncertainty: ${JSON.stringify(profile?.uncertainty ?? [])}`,
      '</recipient_model>',
      '<participant_models>',
      JSON.stringify(participantModels, null, 2),
      '</participant_models>',
      '<interpretation_contract>',
      '- Distinguish observed facts from modeled beliefs or inferred intent.',
      '- Preserve unresolved uncertainty and disagreement across participants.',
      '- Use only observable trace content; do not infer hidden chain-of-thought.',
      '</interpretation_contract>',
      base.rendered,
      '</agent_communication>',
    ].join('\n');

    return {
      ...base,
      protocolId: this.id,
      protocolVersion: this.version,
      rendered,
      metadata: {
        ...base.metadata,
        recipientToMLevel: profile?.level ?? 0,
        modeledParticipants: participantModels.map(item => item.id),
      },
    };
  }
}
