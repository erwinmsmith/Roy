// Conversational agent prompt template

export const CONVERSATIONAL_AGENT_TEMPLATE = `{agent_identity}
The model provider is only your inference backend and must not be presented as your identity.

You have the following goals:
{agent_goal}
{agent_example}
Now, please follow the goals to chat with the user.
Here is the history of the conversation:
`;

export const conversationalTemplate = {
  name: 'conversational',
  description: 'Template for conversational agent',
  template: CONVERSATIONAL_AGENT_TEMPLATE,
};
