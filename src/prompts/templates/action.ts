// Action agent prompt template

export const ACTION_AGENT_TEMPLATE = `You are an AI Agent. You have the following goals:
{agent_goal}

Please follow the goals to complete the task. You can choose to do the following actions:
{agent_actions}

There is some examples:
{agent_example}

Now, for a new observation, please select the action.
Here is the observation:
`;

export const ACTION_TEMPLATE = `You must Generate a json string. The json string should include the action you choose to do and the parameters.
The parameters of the {action} includes:
{action_params}
You need to follow the following rules to generate the parameters:
{action_rules}
Here is an example:
{action_example}
Now, for a new observation, please generate the json string, remember that the output must be a json string including the action name and the parameters.:
`;

export const actionTemplate = {
  name: 'action',
  description: 'Template for action-based agent',
  template: ACTION_AGENT_TEMPLATE,
};

export const actionCallTemplate = {
  name: 'action_call',
  description: 'Template for generating action parameters',
  template: ACTION_TEMPLATE,
};