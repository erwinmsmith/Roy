#!/usr/bin/env node
// Roy CLI - Terminal Interface for the Agent System

import * as readline from 'readline';
import * as path from 'path';
import { bootstrap, cleanup, type BootstrapContext } from '../bootstrap.js';
import { runtime, type AgentTreeNode, type RuntimeActorNode, type SubAgentArchetype } from '../core/runtime/Runtime.js';
import { skillRegistry } from '../core/skills/index.js';
import { actionRegistry } from '../core/actions/index.js';
import { toolRegistry } from '../core/tools/index.js';
import { logger } from '../core/utils/logger.js';

// ASCII Banner - compatible with all terminals
const BANNER = `
+=====================================================+
|                                                    |
|     ██████╗  ██████╗ ██╗   ██╗                     |
|     ██╔══██╗██╔═══██╗╚██╗ ██╔╝                     |
|     ██████╔╝██║   ██║ ╚████╔╝                      |
|     ██╔══██╗██║   ██║  ╚██╔╝                       |
|     ██║  ██║╚██████╔╝   ██║                        |
|     ╚═╝  ╚═╝ ╚═════╝    ╚═╝                        |
|                                                    |
| Theory of Mind based Autonomous Agent System       |
|                                                    |
+=====================================================+
`;

interface DialogOption {
  key: string;
  label: string;
  description: string;
}

export class Roy {
  private ctx: BootstrapContext | null = null;
  private rl: readline.Interface | null = null;
  private shuttingDown = false;
  private autoColor = true;
  private verboseMode = false;
  private sessionId = process.env.ROY_SESSION_ID ?? `cli-${new Date().toISOString().replace(/[:.]/g, '-')}`;

  // Color utilities
  private green(text: string): string {
    return this.autoColor ? `\x1b[32m${text}\x1b[0m` : text;
  }

  private yellow(text: string): string {
    return this.autoColor ? `\x1b[33m${text}\x1b[0m` : text;
  }

  private red(text: string): string {
    return this.autoColor ? `\x1b[31m${text}\x1b[0m` : text;
  }

  private cyan(text: string): string {
    return this.autoColor ? `\x1b[36m${text}\x1b[0m` : text;
  }

  private bold(text: string): string {
    return this.autoColor ? `\x1b[1m${text}\x1b[0m` : text;
  }

  private dim(text: string): string {
    return this.autoColor ? `\x1b[2m${text}\x1b[0m` : text;
  }

  async launch(): Promise<void> {
    this.printBanner();

    try {
      this.ctx = await bootstrap({
        agentName: 'Roy',
        agentGoal: 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
        sessionId: this.sessionId,
        fsmEnabled: true,
      });

      logger.info('CLI Bootstrap complete');
      await this.printReady();
      this.rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        completer: this.completer.bind(this),
      });
      await this.startChat();
    } catch (error) {
      console.log('\n  ' + this.red('[ERROR]') + ' Failed to initialize Roy');
      console.log('  ' + this.dim(String(error)));
      logger.error('Bootstrap failed:', error);
      process.exitCode = 1;
    }
  }

  private printBanner(): void {
    console.log(this.green(BANNER));
    console.log(this.dim('='.repeat(60)));
  }

  private async printReady(): Promise<void> {
    if (!this.ctx) return;

    const state = runtime.getState();
    const memory = await runtime.getMemoryState();
    const provider = this.ctx.llm?.name ?? 'not configured';
    const model = this.ctx.llm?.defaultModel ?? this.ctx.config.llm?.model ?? 'unknown';
    const budget = state.budget.mode === 'unlimited'
      ? 'unlimited'
      : `${state.budget.usedTokens} / ${state.budget.limitTokens} tokens used`;

    console.log(this.green('Roy Runtime initialized'));
    console.log('Provider: ' + this.cyan(provider));
    console.log('Model: ' + this.cyan(model));
    console.log('Root Agent: ' + this.cyan(`${state.rootAgent.identity.name} [${state.rootAgent.identity.role}, ToM-${state.rootAgent.identity.tomLevel}]`));
    console.log('Session: ' + this.cyan(state.sessionId));
    console.log('FSM: ' + this.cyan(this.ctx.fsm.getStateName()));
    console.log('Budget: ' + this.cyan(budget));
    console.log('Tokens: ' + this.cyan(`${state.budget.usedTokens} total`));
    console.log('Workspace: ' + this.cyan(path.relative(process.cwd(), memory.rootPath) || memory.rootPath));
    console.log('Memory: ' + this.cyan(`${memory.memoryDocs.length} docs loaded`));
    console.log('Patterns: ' + this.cyan(`${memory.patterns.agents} agents, ${memory.patterns.teams} teams`));
    console.log('API: ' + this.dim('http://localhost:' + (this.ctx.config.server?.port ?? 3000)));
    console.log('Type ' + this.cyan('/help') + ' for commands.');
  }

  private async showWelcomeDialog(): Promise<void> {
    return new Promise((resolve) => {
      console.log('\n' + this.bold('  Welcome to Roy - Your Agent System Terminal') + '\n');

      const options: DialogOption[] = [
        { key: '1', label: 'Quick Start', description: 'Start with default settings' },
        { key: '2', label: 'Configure LLM', description: 'Check/configure LLM provider' },
        { key: '3', label: 'View Capabilities', description: 'Show skills, actions, tools' },
        { key: '4', label: 'Advanced', description: 'Configure all settings' },
      ];

      console.log('  ' + this.bold('Choose an option:'));
      console.log('');

      for (const opt of options) {
        console.log(`    [${this.cyan(opt.key)}] ${this.bold(opt.label)} - ${opt.description}`);
      }

      console.log('');

      const question = '\n  Press Enter to start with defaults, or type option number: ';
      this.requireReadline().question(question, async (answer) => {
        const choice = answer.trim();

        if (choice === '2') {
          await this.showLLMConfigDialog();
        } else if (choice === '3') {
          await this.showCapabilities();
          await this.pause('Press Enter to continue...');
        } else if (choice === '4') {
          await this.showAdvancedConfigDialog();
        }

        resolve();
      });
    });
  }

  private async showLLMConfigDialog(): Promise<void> {
    console.log('\n' + this.bold('  LLM Configuration') + '\n');

    const provider = this.ctx?.config.llm?.provider ?? 'anthropic';
    const model = this.ctx?.config.llm?.model ?? 'claude-sonnet-4-20250514';
    const apiKeySet = !!(process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || this.ctx?.config.llm?.apiKey);

    console.log(`    Provider: ${this.cyan(provider)}`);
    console.log(`    Model:    ${this.cyan(model)}`);
    console.log(`    API Key:  ${apiKeySet ? this.green('Configured') : this.red('Not set')}`);
    console.log('');

    if (!apiKeySet) {
      console.log('  ' + this.yellow('No API key detected. Please set:'));
      console.log('    ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable');
      console.log('    Or configure in roy.config.yaml');
    }
  }

  private async showAdvancedConfigDialog(): Promise<void> {
    console.log('\n' + this.bold('  Advanced Configuration') + '\n');

    const config = this.ctx?.config;
    const configInfo = [
      ['Port', String(config?.server?.port ?? 3000)],
      ['Log Level', config?.logger?.level ?? 'info'],
      ['LLM Provider', config?.llm?.provider ?? 'auto'],
      ['LLM Model', config?.llm?.model ?? 'default'],
    ];

    for (const [key, value] of configInfo) {
      console.log(`    ${this.bold(key + ':')} ${this.cyan(value)}`);
    }

    console.log('\n  ' + this.dim('Configure via roy.config.yaml for persistent settings'));
  }

  private async showCapabilities(): Promise<void> {
    console.log('\n' + this.bold('  System Capabilities') + '\n');

    const caps = this.ctx?.capabilities;

    console.log(`    ${this.bold('Skills:')} ${caps?.skills ?? 0}`);
    const skills = skillRegistry.list();
    if (skills.length > 0) {
      for (const skill of skills.slice(0, 5)) {
        console.log(`      - ${skill.name}`);
      }
      if (skills.length > 5) {
        console.log(`      ${this.dim('... and ' + (skills.length - 5) + ' more')}`);
      }
    }

    console.log(`    ${this.bold('Actions:')} ${caps?.actions ?? 0}`);
    const actions = actionRegistry.list();
    if (actions.length > 0) {
      for (const action of actions.slice(0, 5)) {
        console.log(`      - ${action.name}`);
      }
      if (actions.length > 5) {
        console.log(`      ${this.dim('... and ' + (actions.length - 5) + ' more')}`);
      }
    }

    console.log(`    ${this.bold('Tools:')} ${caps?.tools ?? 0}`);
  }

  private async pause(message: string): Promise<void> {
    return new Promise((resolve) => {
      this.requireReadline().question('\n  ' + message + ' ', () => {
        resolve();
      });
    });
  }

  private printStatus(): void {
    if (!this.ctx) return;

    console.log(this.dim('-'.repeat(60)));
    console.log('  ' + this.bold('Status'));
    console.log('    LLM:       ' + (this.ctx.llm ? this.green('Connected') : this.red('Not configured')));
    const agentInfo = this.ctx.agent.getInfo();
    const usage = agentInfo.usage;
    console.log('    Agent:     ' + this.cyan(agentInfo.name) + ' ' + this.green(`[${agentInfo.role}]`) + ' ' + this.green('[active]'));
    console.log('    FSM:       ' + this.cyan(this.ctx.fsm.getStateName()));
    console.log('    Session:   ' + this.cyan(this.sessionId));
    console.log('    Memory:    ' + this.dim(this.getMemoryStats()));

    const fsmInfo = this.ctx.agent.getFSMInfo();
    if (fsmInfo) {
      const budget = fsmInfo.budget === null ? 'unlimited' : String(fsmInfo.budget);
      console.log('    Budget:    ' + budget + ', Cost: ' + fsmInfo.cost);
    }
    console.log('    Tokens:    ' + `${usage.totalTokens} total (${usage.promptTokens} prompt, ${usage.completionTokens} completion), ${usage.llmCalls} calls`);

    console.log('    API Mode:  ' + this.dim('http://localhost:' + (this.ctx.config.server?.port ?? 3000)));
    console.log(this.dim('-'.repeat(60)));
    console.log('');
  }

  private getMemoryStats(): string {
    if (!this.ctx) return 'N/A';
    const agent = this.ctx.agent;
    const messages = agent.getRecentMessages(100);
    return messages.length + ' messages';
  }

  private async startChat(): Promise<void> {
    const rl = this.requireReadline();
    const interactive = Boolean(process.stdin.isTTY && process.stdout.isTTY);
    if (interactive) {
      rl.setPrompt(this.cyan('\nyou') + ' > ');
      rl.prompt();
    }
    try {
      for await (const input of rl) {
        const trimmed = input.trim();
        if (trimmed) {
          if (trimmed.startsWith('/')) {
            try {
              const shouldContinue = await this.handleCommand(trimmed);
              if (shouldContinue === false) break;
            } catch (error) {
              console.log('\n  ' + this.red('Command error:') + ' ' + (error instanceof Error ? error.message : String(error)) + '\n');
            }
          } else {
            await this.processMessage(trimmed);
          }
        }
        if (interactive) rl.prompt();
      }
    } finally {
      rl.close();
      await this.shutdown();
    }
  }

  private requireReadline(): readline.Interface {
    if (!this.rl) throw new Error('CLI input is not initialized');
    return this.rl;
  }

  private async shutdown(): Promise<void> {
    if (this.shuttingDown) return;
    this.shuttingDown = true;
    console.log('\n\n' + this.yellow('Goodbye! Roy shutting down...') + '\n');
    if (this.ctx) await cleanup(this.ctx);
  }

  private async handleCommand(command: string): Promise<boolean | undefined> {
    const parts = this.parseCommand(command);
    const cmd = parts[0]?.toLowerCase();
    const args = parts.slice(1).join(' ');

    switch (cmd) {
      case '/help':
      case '/h':
        this.printHelp();
        break;

      case '/clear':
      case '/cls':
        console.log('\n  ' + this.green('Note: Memory is managed by the agent. Use /reset to clear.') + '\n');
        break;

      case '/reset':
        this.ctx?.fsm.reset();
        console.log('\n  ' + this.green('FSM reset to initial state') + '\n');
        break;

      case '/agents':
        if (parts[1] === 'archetypes') {
          this.printAgentArchetypes();
        } else if (parts[1] === 'policy') {
          this.printAgentPolicy(parts[2]);
        } else {
          this.printAgents(parts.includes('--tree'), parts.includes('--tom'));
        }
        break;

      case '/spawn':
        await this.spawnAgent(parts);
        break;

      case '/run':
        await this.runAgent(parts);
        break;

      case '/teams':
        this.printTeams(parts.includes('--tree'), parts.includes('--tom'));
        break;

      case '/tom':
        this.printToMState(parts[1]);
        break;

      case '/team':
        await this.handleTeam(parts);
        break;

      case '/exit':
      case '/quit':
      case '/q':
        return false;

      case '/api':
        this.printApiInfo();
        break;

      case '/budget':
        this.handleBudget(args);
        break;

      case '/events':
        this.printEvents(parts);
        break;

      case '/queue':
        await this.printQueue();
        break;

      case '/cache':
        await this.printCache(parts);
        break;

      case '/messages':
        await this.printMessages(parts);
        break;

      case '/traces':
        await this.printTraces(parts);
        break;

      case '/config':
        await this.showAdvancedConfigDialog();
        break;

      case '/status':
        this.printStatus();
        break;

      case '/skills':
        this.printSkills();
        break;

      case '/actions':
        this.printActions();
        break;

      case '/tools':
        await this.handleTools(parts);
        break;

      case '/memory':
        await this.printWorkspaceMemory(args);
        break;

      case '/session':
        this.printSession();
        break;

      case '/verbose':
        this.verboseMode = !this.verboseMode;
        console.log('\n  Verbose mode: ' + (this.verboseMode ? this.green('on') : this.dim('off')) + '\n');
        break;

      case '/color':
        this.autoColor = !this.autoColor;
        console.log('\n  Colors: ' + (this.autoColor ? this.green('on') : this.dim('off')) + '\n');
        break;

      case '/system':
        this.printSystemInfo();
        break;

      case '/fsm':
        this.printFSMInfo();
        break;

      case '/prompt':
        if (parts[1] === 'render') {
          await this.renderPrompt(parts);
        } else if (parts[1] === 'agent') {
          await this.printAgentPrompt(parts[2] ?? 'roy');
        } else if (args && this.ctx) {
          this.ctx.agent.addToMemory('meta', `System prompt: ${args}`);
          console.log('\n  ' + this.green('System prompt added to agent memory') + '\n');
        } else {
          console.log('\n  Usage: /prompt agent <agentKey> | /prompt render <agentKey> --task "..." | /prompt <system instructions>' + '\n');
        }
        break;

      case '/context':
        await this.handleContext(parts);
        break;

      case '/conversation':
        await this.printConversation(parts);
        break;

      default:
        console.log('\n  ' + this.red('Unknown command:') + ' ' + command);
        console.log('  Type ' + this.cyan('/help') + ' for available commands\n');
    }
  }

  private printHelp(): void {
    console.log(`
  ${this.bold('Available Commands:')}

    ${this.bold('Chat & History')}
      /clear, /cls        Clear (note: history managed by agent)
      /context            Show conversation context from memory
      /context render <agent> --task "..." Show compact context sources and token usage
      /traces             Show persisted trace files
      /traces latest      Show latest persisted trace events

    ${this.bold('System Information')}
      /help, /h           Show this help message
      /status             Show connection status with FSM state
      /system             Show system information
      /fsm                Show FSM state and trace
      /budget             Show token budget and usage
      /budget set <n>     Set token budget
      /budget unlimited   Remove token budget limit
      /budget market      Show budget allocations and reservations
      /events             Show recent runtime events
      /events --agent <id> Filter events by agent
      /events --type <type> Filter events by event type
      /events --latest <n> Show latest N events
      /queue              Show runtime message queue state
      /cache agents       Show cached agent patterns
      /cache delegations  Show cached delegation patterns
      /cache teams        Show cached team patterns
      /cache evolution    Show delegation evolution history
      /messages --correlation <id> Show messages for a delegation chain
      /memory             Show workspace memory state
      /memory public      Show public memory docs
      /memory public <doc> Show project, context, decisions, constraints, glossary, or user
      /memory agent <key> Show agent memory, prompt, and context
      /memory proposals   Show pending memory update proposals
      /memory show <id>   Show proposal content before accepting
      /memory signals     Show parsed memory signals from the current session
      /memory accept <id> Commit a memory proposal
      /memory reject <id> Reject a memory proposal
      /memory summarize   Generate memory proposals from the current session
      /memory mode <mode> Set memory mode: suggest, auto-safe, or off
      /memory auto on|off Enable or disable auto-propose
      /conversation       Show persisted conversation log
      /conversation sessions List persisted conversation sessions
      /conversation --session <id> Show a specific session
      /conversation import <path> Import JSON/JSONL conversation
      /verbose            Toggle verbose mode

    ${this.bold('Agent Management')}
      /agents             List available agents
      /agents --tree      Show agent parent-child tree
      /agents --tree --tom Show agent tree with ToM profiles
      /agents archetypes  Show built-in archetype skills/tools
      /agents policy <id> Show spawn policy for an agent
      /tom [correlation-id] Show cognitive gaps and ToM profiles
      /spawn <type> "task" Spawn and run a controlled subagent
      /spawn <type> --parent <id> "task" Spawn below another agent
      /spawn custom --name <name> [--role <role>] [--style <style>] "task"
      /run <agent-id> "task" Run an existing subagent
      /session            Show current session info
      /reset              Reset FSM to initial state

    ${this.bold('Capabilities')}
      /skills             List registered skills
      /actions            List available actions
      /tools              List available tools
      /tools approvals    List pending tool approvals
      /tools approve <id> Approve a pending tool request
      /tools deny <id>    Deny a pending tool request
      /teams              Show runtime subteams
      /teams --tree       Show root, team, and member actor tree
      /teams --tree --tom Show actor tree with team/agent ToM details
      /team <team-id>     Show one team and its member tree
      /team create --name <name> --description <text> [--mode sequential|parallel] [--failure fail_fast|best_effort]
      /team add <team-id> <archetype> "task"
      /team run <team-id> "task"
      /memory             Show workspace memory and agent memory statistics

    ${this.bold('Configuration')}
      /api                Show API information
      /config             Show runtime configuration
      /prompt agent <key> Show raw agent prompt.md
      /prompt render <key> --task "..." Render final system prompt preview
      /context render <key> --task "..." Render context/prompt sources
      /color              Toggle color output

    ${this.bold('Exit')}
      /exit, /quit, /q    Exit Roy
`);
  }

  private printAgents(tree = false, showTom = false): void {
    if (!this.ctx) return;

    if (tree) {
      console.log('\n  ' + this.bold('Agent Tree:'));
      this.printAgentTree(runtime.getAgentTree(), '    ', true, showTom);
      console.log('');
      return;
    }

    const agents = this.ctx.manager.listAgentInfo();
    console.log('\n  ' + this.bold('Agents:'));
    if (agents.length === 0) {
      console.log('    ' + this.dim('No agents registered'));
    } else {
      for (const agent of agents) {
        const isActive = agent.name === this.ctx.agent.name;
        const usage = agent.usage;
        const parent = agent.identity.parentId ?? '-';
        console.log(`    - ${this.cyan(agent.identity.id)} ${agent.name} ${this.dim(agent.role)} ${isActive ? this.green('[active]') : ''}`);
        console.log(`      state=${agent.state}, tom=${agent.identity.tomLevel}, tokens=${usage.totalTokens}, calls=${usage.llmCalls}, parent=${parent}`);
        if (showTom) {
          this.printToMProfile(agent.identity.tomProfile, '      ');
        }
      }
    }
    console.log('');
  }

  private printAgentArchetypes(): void {
    console.log('\n  ' + this.bold('Agent Archetypes'));
    for (const profile of runtime.getAgentArchetypeProfiles()) {
      const tools = profile.tools.map(tool => tool.name).join(', ') || 'none';
      const skills = profile.skills.map(skill => skill.name).join(', ') || 'none';
      console.log(`    - ${this.cyan(profile.archetype)}`);
      console.log(`      tools:  ${tools}`);
      console.log(`      skills: ${skills}`);
      console.log(`      spawn:  maxChildren=${profile.spawnPolicy.maxChildren}, maxDepth=${profile.spawnPolicy.maxDepth}, custom=${profile.spawnPolicy.allowCustomAgents ? 'allowed' : 'blocked'}`);
    }
    console.log('');
  }

  private printAgentPolicy(agentId: string | undefined): void {
    if (!agentId) {
      console.log('\n  Usage: /agents policy <agentId>\n');
      return;
    }
    const policy = runtime.getAgentPolicy(agentId);
    if (!policy) {
      console.log('\n  ' + this.yellow(`No agent found: ${agentId}`) + '\n');
      return;
    }
    console.log('\n  ' + this.bold(`Spawn Policy for ${agentId}`));
    console.log(`    parent:          ${policy.parentId ?? '-'}`);
    console.log(`    depth:           ${policy.depth}`);
    console.log(`    canSpawn:        ${policy.spawnPolicy.canSpawn ? 'yes' : 'no'}`);
    console.log(`    maxChildren:     ${policy.spawnPolicy.maxChildren}`);
    console.log(`    currentChildren: ${policy.currentChildren}`);
    console.log(`    allowedChildren: ${policy.allowedChildren}`);
    console.log(`    maxDepth:        ${policy.spawnPolicy.maxDepth}`);
    console.log(`    customAgents:    ${policy.spawnPolicy.allowCustomAgents ? 'allowed' : 'blocked'}`);
    console.log(`    budgetAware:     ${policy.spawnPolicy.budgetAware ? 'yes' : 'no'}`);
    console.log(`    allowedStates:   ${policy.spawnPolicy.allowedStates.join(', ')}`);
    console.log(`    tools:           ${policy.tools.map(tool => tool.name).join(', ') || 'none'}`);
    console.log(`    skills:          ${policy.skills.map(skill => skill.name).join(', ') || 'none'}`);
    console.log('');
  }

  private printAgentTree(node: ReturnType<typeof runtime.getAgentTree>, prefix: string, isRoot = false, showTom = false): void {
    const agent = node.agent;
    const usage = agent.usage;
    const label = `${agent.name} [${agent.role}, ToM-${agent.identity.tomProfile.level}, ${agent.state}, ${usage.totalTokens} tokens]`;
    console.log(prefix + (isRoot ? '' : '└── ') + this.cyan(label));
    if (showTom) {
      this.printToMProfile(agent.identity.tomProfile, prefix + (isRoot ? '  ' : '    '));
    }
    const childPrefix = prefix + (isRoot ? '' : '    ');
    for (let i = 0; i < node.children.length; i++) {
      this.printAgentTree(node.children[i], childPrefix, false, showTom);
    }
  }

  private printApiInfo(): void {
    console.log('\n  ' + this.bold('API Endpoints:'));
    console.log('    GET  /           - Server info');
    console.log('    GET  /health     - Health check');
    console.log('    POST /v1/chat    - Root-controlled chat/delegation turn');
    console.log('    GET  /v1/status  - Runtime status');
    console.log('    GET  /v1/agents  - Agent states');
    console.log('    GET  /v1/agents/tree - Agent tree');
    console.log('    GET  /v1/tom     - ToM analyses, gaps, and actor profiles');
    console.log('    POST /v1/agents  - Spawn subagent');
    console.log('    POST /v1/agents/:id/run - Run subagent');
    console.log('    GET  /v1/teams - Team states');
    console.log('    GET  /v1/teams/tree - Team actor tree');
    console.log('    POST /v1/teams - Create subteam');
    console.log('    POST /v1/teams/:id/run - Run subteam');
    console.log('    GET  /v1/runtime/sessions - Active HTTP runtime sessions');
    console.log('    DELETE /v1/runtime/session - Close the selected HTTP runtime session');
    console.log('    GET  /v1/budget  - Token budget');
    console.log('    GET  /v1/events  - Runtime events');
    console.log('    GET  /v1/queue   - Runtime message queue');
    console.log('    GET  /v1/memory  - Workspace memory state');
    console.log('    GET  /v1/memory/root - Root memory context');
    console.log('    WS   /           - Socket.IO for real-time chat');
    console.log('    Port: ' + this.cyan(String(this.ctx?.config.server?.port ?? 3000)));
    console.log('');
  }

  private printBudget(): void {
    if (!this.ctx) return;

    const budget = runtime.getBudgetState();
    console.log('\n  ' + this.bold('Budget'));
    if (budget.mode === 'unlimited') {
      console.log('    Budget: unlimited');
      console.log(`    Session used: ${budget.usedTokens} tokens`);
    } else {
      console.log(`    Budget: ${budget.usedTokens} / ${budget.limitTokens} tokens`);
      console.log(`    Remaining: ${budget.remainingTokens ?? 0} tokens`);
    }
    console.log('    Thinking Tokens: unavailable');
    console.log('\n  ' + this.bold('Per Agent:'));
    for (const agent of runtime.getState().agents) {
      const total = budget.perAgent[agent.identity.id]?.totalTokens ?? 0;
      console.log(`    ${agent.name.padEnd(18)} ${String(total).padStart(8)} tokens`);
    }
    console.log('\n  ' + this.bold('Per Team:'));
    const teams = runtime.getTeams();
    if (teams.length === 0) console.log('    ' + this.dim('No runtime teams'));
    for (const team of teams) {
      const usage = budget.perTeam[team.identity.id] ?? team.tokenUsage;
      console.log(`    ${team.identity.name.padEnd(18)} ${String(usage.totalTokens).padStart(8)} tokens`);
      for (const agentId of team.memberAgentIds) {
        const member = runtime.getState().agents.find(agent => agent.identity.id === agentId);
        const memberTokens = team.memberUsage[agentId]?.totalTokens ?? 0;
        console.log(`      ${member?.identity.name ?? agentId} ${memberTokens} tokens`);
      }
      console.log(`      team synthesis ${team.synthesisUsage.totalTokens} tokens`);
    }
    console.log('');
  }

  private handleBudget(args: string): void {
    const [subcommand, value] = args.split(/\s+/);
    if (subcommand === 'set') {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        console.log('\n  Usage: /budget set <positive-token-limit>\n');
        return;
      }
      runtime.setBudget(parsed);
      console.log('\n  ' + this.green(`Budget set to ${parsed} tokens`) + '\n');
      return;
    }

    if (subcommand === 'unlimited' || subcommand === 'clear') {
      runtime.setBudget(null);
      console.log('\n  ' + this.green('Budget set to unlimited') + '\n');
      return;
    }

    if (subcommand === 'market') {
      const market = runtime.getBudgetMarketState();
      console.log('\n  ' + this.bold('Budget Market'));
      console.log(`    Mode:       ${market.mode}`);
      console.log(`    Used:       ${market.usedTokens}`);
      console.log(`    Reserved:   ${market.reservedTokens}`);
      console.log(`    Available:  ${market.availableTokens ?? 'unlimited'}`);
      for (const allocation of market.allocations.slice(-10)) {
        console.log(`    - ${this.cyan(allocation.id)} ${allocation.status} ${allocation.grantedTokens}/${allocation.request.requestedTokens} ${allocation.request.purpose}`);
      }
      console.log('');
      return;
    }

    this.printBudget();
  }

  private printEvents(parts: string[] = []): void {
    const agentIndex = parts.indexOf('--agent');
    const typeIndex = parts.indexOf('--type');
    const latestIndex = parts.indexOf('--latest');
    const agentId = agentIndex >= 0 ? parts[agentIndex + 1] : undefined;
    const eventType = typeIndex >= 0 ? parts[typeIndex + 1] : undefined;
    const latest = latestIndex >= 0 ? Number(parts[latestIndex + 1]) : 10;
    const limit = Number.isFinite(latest) && latest > 0 ? latest : 10;
    const events = runtime.getEvents()
      .filter(event => !agentId || event.agentId === agentId)
      .filter(event => !eventType || event.type === eventType)
      .slice(-limit);
    const label = agentId ? `Events for ${agentId}` : eventType ? `Events: ${eventType}` : 'Recent Events';
    console.log('\n  ' + this.bold(label));
    if (events.length === 0) {
      console.log('    ' + this.dim('No events'));
    } else {
      for (const event of events) {
        console.log(`    - ${this.dim(new Date(event.timestamp).toISOString())} ${this.cyan(event.type)}${event.agentId ? ' ' + event.agentId : ''}`);
      }
    }
    console.log('');
  }

  private async printTraces(parts: string[]): Promise<void> {
    if (parts[1] === 'latest' || parts[1] === 'show') {
      const name = parts[1] === 'show' ? parts[2] ?? 'latest' : 'latest';
      const limit = Number(parts[3] ?? 30);
      const events = await runtime.readTrace(name, Number.isFinite(limit) && limit > 0 ? limit : 30);
      console.log('\n  ' + this.bold(`Trace: ${name}`));
      if (events.length === 0) {
        console.log('    ' + this.dim('No trace events'));
      } else {
        for (const event of events) {
          console.log(`    - ${this.dim(new Date(event.timestamp).toISOString())} ${this.cyan(event.type)}${event.agentId ? ' ' + event.agentId : ''}`);
        }
      }
      console.log('');
      return;
    }

    const traces = await runtime.listTraces();
    console.log('\n  ' + this.bold('Traces'));
    if (traces.length === 0) {
      console.log('    ' + this.dim('No persisted traces'));
    } else {
      for (const trace of traces) {
        console.log(`    - ${this.cyan(trace.name)} ${this.dim(`${trace.size} bytes`)} ${this.dim(new Date(trace.updatedAt).toISOString())}`);
      }
    }
    console.log('');
  }

  private async printQueue(): Promise<void> {
    const state = await runtime.getQueueState(10);
    console.log('\n  ' + this.bold('Queue'));
    console.log(`    Pending:     ${state.stats.pending}`);
    console.log(`    Processing:  ${state.stats.processing}`);
    console.log(`    Completed:   ${state.stats.completed}`);
    console.log(`    Failed:      ${state.stats.failed}`);
    console.log(`    Cancelled:   ${state.stats.cancelled}`);

    console.log('\n  ' + this.bold('Recent:'));
    if (state.recent.length === 0) {
      console.log('    ' + this.dim('No messages'));
    } else {
      for (const message of state.recent) {
        console.log(`    ${message.id.slice(0, 8)} ${message.kind.padEnd(14)} ${message.from} -> ${message.to} ${this.dim(message.status)}`);
      }
    }
    console.log('');
  }

  private printSkills(): void {
    const skills = skillRegistry.list();
    console.log('\n  ' + this.bold('Registered Skills:'));
    if (skills.length === 0) {
      console.log('    ' + this.dim('No skills registered'));
    } else {
      for (const skill of skills) {
        const manifest = skillRegistry.getManifest(skill.name);
        console.log(`    - ${this.cyan(skill.name)}`);
        if (manifest?.description) {
          console.log(`      ${this.dim(manifest.description)}`);
        }
        if (manifest?.tags?.length) {
          console.log(`      Tags: ${manifest.tags.join(', ')}`);
        }
      }
    }
    console.log('');
  }

  private printActions(): void {
    const actions = actionRegistry.list();
    console.log('\n  ' + this.bold('Available Actions:'));
    if (actions.length === 0) {
      console.log('    ' + this.dim('No actions registered'));
    } else {
      for (const action of actions) {
        console.log(`    - ${this.cyan(action.name)}: ${action.description}`);
        if (action.parameters?.length) {
          console.log(`      Parameters: ${action.parameters.map(p => p.name).join(', ')}`);
        }
      }
    }
    console.log('');
  }

  private printTools(): void {
    const tools = toolRegistry.list();
    console.log('\n  ' + this.bold('Available Tools:'));
    if (tools.length === 0) {
      console.log('    ' + this.dim('No tools registered'));
    } else {
      for (const tool of tools) {
        console.log(`    - ${this.cyan(tool.name)}: ${tool.description}`);
      }
    }
    console.log('');
  }

  private async handleTools(parts: string[]): Promise<void> {
    const command = parts[1];
    if (command === 'approvals') {
      const approvals = runtime.getToolApprovals();
      console.log('\n  ' + this.bold('Tool Approvals'));
      if (approvals.length === 0) console.log('    ' + this.dim('No tool approval requests'));
      for (const approval of approvals) {
        console.log(`    - ${this.cyan(approval.id)} ${approval.status} ${approval.agentId} -> ${approval.toolName} (${approval.permission})`);
      }
      console.log('');
      return;
    }
    if ((command === 'approve' || command === 'deny') && parts[2]) {
      const resolved = await runtime.resolveToolApproval(parts[2], command === 'approve' ? 'approved' : 'denied');
      console.log(resolved
        ? `\n  ${this.green(`Tool request ${resolved.id} ${resolved.status}`)}\n`
        : `\n  ${this.red(`Pending tool request not found: ${parts[2]}`)}\n`);
      return;
    }
    this.printTools();
  }

  private printTeams(tree = false, showTom = false): void {
    const teams = runtime.getTeams();
    console.log('\n  ' + this.bold('Runtime Teams'));
    if (teams.length === 0) {
      console.log('    ' + this.dim('No runtime teams'));
    } else if (tree) {
      const actorTree = runtime.getTeamActorTree();
      this.printRuntimeActorTree(actorTree.hierarchy, '    ', true, true, showTom);
    } else {
      for (const team of teams) {
        console.log(`    - ${this.cyan(team.identity.id)} ${team.identity.name} [${team.status}] parent=${team.identity.parentAgentId}`);
        console.log(`      fsm=${team.fsmState} members=${team.memberAgentIds.length} tokens=${team.tokenUsage.totalTokens}`);
        console.log(`      execution=${team.executionPolicy.mode}/${team.executionPolicy.failureMode} concurrency=${team.executionPolicy.maxConcurrency} minSuccess=${team.executionPolicy.minimumSuccessfulMembers}`);
        if (showTom) this.printToMProfile(team.identity.tomProfile, '      ');
        const failedMembers = Object.entries(team.memberErrors);
        if (failedMembers.length > 0) console.log(`      failures=${failedMembers.map(([id, error]) => `${id}: ${error}`).join('; ')}`);
      }
    }
    console.log('');
  }

  private printTeamMemberTree(node: AgentTreeNode, prefix: string, last: boolean): void {
    const branch = last ? '└── ' : '├── ';
    const continuation = last ? '    ' : '│   ';
    console.log(`${prefix}${branch}${node.agent.identity.name} [${node.agent.identity.role}, ${node.agent.state}, ${node.agent.usage.totalTokens} tokens]`);
    node.children.forEach((child, index) => {
      this.printTeamMemberTree(child, prefix + continuation, index === node.children.length - 1);
    });
  }

  private printRuntimeActorTree(node: RuntimeActorNode, prefix: string, last: boolean, root = false, showTom = false): void {
    const branch = root ? '' : last ? '└── ' : '├── ';
    const continuation = root ? '' : last ? '    ' : '│   ';
    if (node.type === 'agent') {
      console.log(`${prefix}${branch}${this.cyan(node.agent.identity.name)} [${node.agent.identity.role}, ${node.agent.state}, ${node.agent.usage.totalTokens} tokens]`);
    } else {
      console.log(`${prefix}${branch}${this.cyan(node.team.identity.name)} [subteam, ToM-${node.team.identity.tomLevel}, ${node.team.status}, ${node.team.tokenUsage.totalTokens} tokens]`);
    }
    if (showTom) {
      const profile = node.type === 'agent' ? node.agent.identity.tomProfile : node.team.identity.tomProfile;
      this.printToMProfile(profile, prefix + continuation + '  ');
    }
    node.children.forEach((child, index) => {
      this.printRuntimeActorTree(child, prefix + continuation, index === node.children.length - 1, false, showTom);
    });
  }

  private printToMProfile(profile: ReturnType<typeof runtime.getState>['rootAgent']['identity']['tomProfile'], prefix: string): void {
    const line = (label: string, values: string[] | string | undefined): void => {
      const value = Array.isArray(values) ? values.join('; ') : values;
      console.log(`${prefix}${this.dim(`${label}: ${value?.trim() || 'none'}`)}`);
    };
    line('purpose', profile.purpose);
    line('perspective', profile.perspective);
    line('belief', profile.beliefScope);
    line('goal', profile.goalModel);
    line('uncertainty', profile.uncertainty);
    line('observes', profile.observesAgents);
    line('models', profile.modelsAgents.length > 0 ? profile.modelsAgents : profile.models.map(model => model.targetId));
    line('gaps', profile.cognitiveGaps);
  }

  private printToMState(correlationId?: string): void {
    const state = runtime.getToMState(correlationId);
    console.log('\n  ' + this.bold('Theory of Mind State'));
    if (state.analyses.length === 0) {
      console.log('    ' + this.dim(correlationId ? `No analysis found for ${correlationId}` : 'No delegation analyses recorded yet'));
    }
    for (const analysis of state.analyses.slice(-5)) {
      console.log(`    - ${this.cyan(analysis.id)} parent=${analysis.parentId} higherOrder=${analysis.requiresHigherOrderToM ? 'yes' : 'no'}`);
      console.log(`      rationale: ${analysis.rationale}`);
      for (const gap of analysis.gaps) {
        console.log(`      ${gap.id} [${gap.kind}, priority=${gap.priority}]`);
        console.log(`        need: ${gap.description}`);
        console.log(`        perspective: ${gap.requiredPerspective}`);
      }
    }
    console.log(`    Actors: ${state.agents.length} agents, ${state.teams.length} teams`);
    console.log('');
  }

  private async handleTeam(parts: string[]): Promise<void> {
    const subcommand = parts[1];
    try {
      if (subcommand === 'create') {
        const name = this.optionValue(parts, '--name');
        const description = this.optionValue(parts, '--description') ?? this.optionValue(parts, '--role');
        if (!name || !description) {
          console.log('\n  Usage: /team create --name <name> --description <description> [--mode sequential|parallel] [--failure fail_fast|best_effort] [--concurrency <n>] [--min-success <n>]\n');
          return;
        }
        const mode = this.optionValue(parts, '--mode');
        const failureMode = this.optionValue(parts, '--failure');
        const maxConcurrency = this.optionValue(parts, '--concurrency');
        const minimumSuccessfulMembers = this.optionValue(parts, '--min-success');
        const team = await runtime.spawnTeam({
          name,
          description,
          parentAgentId: this.optionValue(parts, '--parent') ?? 'root',
          executionPolicy: {
            ...(mode ? { mode: mode as 'sequential' | 'parallel' } : {}),
            ...(failureMode ? { failureMode: failureMode as 'fail_fast' | 'best_effort' } : {}),
            ...(maxConcurrency ? { maxConcurrency: Number(maxConcurrency) } : {}),
            ...(minimumSuccessfulMembers ? { minimumSuccessfulMembers: Number(minimumSuccessfulMembers) } : {}),
          },
        });
        console.log(`\n  ${this.green(`Created ${team.identity.name}`)} ${this.dim(team.identity.id)}\n`);
        return;
      }
      if (subcommand === 'add') {
        const teamId = parts[2];
        const archetype = parts[3] as SubAgentArchetype | undefined;
        const task = parts.slice(4).join(' ').trim();
        if (!teamId || !archetype || !task) {
          console.log('\n  Usage: /team add <team-id> <archetype> "task"\n');
          return;
        }
        const team = await runtime.spawnAgentIntoTeam(teamId, { archetype, task });
        console.log(`\n  ${this.green('Team member planned:')} ${archetype} for ${team.identity.name}. Run ${this.cyan(`/team run ${teamId} "task"`)} to execute.\n`);
        return;
      }
      if (subcommand === 'run') {
        const teamId = parts[2];
        const task = parts.slice(3).join(' ').trim();
        if (!teamId || !task) {
          console.log('\n  Usage: /team run <team-id> "task"\n');
          return;
        }
        const result = await runtime.runTeam(teamId, task);
        console.log(`\n  ${this.green(`${result.team.identity.name}[subteam] >`)} ${result.result}\n`);
        return;
      }

      const teamId = subcommand;
      const tree = teamId ? runtime.getTeamTree(teamId) : undefined;
      if (!tree) {
        console.log(teamId ? `\n  ${this.red(`Team not found: ${teamId}`)}\n` : '\n  Usage: /team <team-id>\n');
        return;
      }
      const team = tree.team;
      console.log('\n  ' + this.bold(`${team.identity.name} [${team.identity.id}]`));
      console.log(`    Parent: ${team.identity.parentAgentId}`);
      console.log(`    Status: ${team.status}`);
      console.log(`    FSM:    ${team.fsmState}`);
      console.log(`    ToM:    ${team.identity.tomLevel}`);
      this.printToMProfile(team.identity.tomProfile, '    ');
      console.log(`    Lead:   ${team.leadAgentId ?? 'none'}`);
      console.log(`    Policy: ${team.executionPolicy.mode}, ${team.executionPolicy.failureMode}, concurrency=${team.executionPolicy.maxConcurrency}, minSuccess=${team.executionPolicy.minimumSuccessfulMembers}`);
      console.log(`    Tokens: ${team.tokenUsage.totalTokens} (members ${team.tokenUsage.totalTokens - team.synthesisUsage.totalTokens}, synthesis ${team.synthesisUsage.totalTokens})`);
      console.log(`    Task:   ${team.task ?? 'none'}`);
      console.log(`    Result: ${team.result ?? 'none'}`);
      const failedMembers = Object.entries(team.memberErrors);
      if (failedMembers.length > 0) {
        console.log('    Member failures:');
        for (const [agentId, error] of failedMembers) console.log(`      - ${agentId}: ${error}`);
      }
      tree.members.forEach((member, index) => {
        this.printTeamMemberTree(member, '    ', index === tree.members.length - 1);
      });
      console.log('');
    } catch (error) {
      console.log(`\n  ${this.red('[ERROR]')} ${error instanceof Error ? error.message : String(error)}\n`);
    }
  }

  private printMemory(): void {
    if (!this.ctx) return;

    console.log('\n  ' + this.bold('Agent Memory:'));
    const messages = this.ctx.agent.getRecentMessages(100);
    console.log(`    Entries: ${this.dim(String(messages.length))}`);

    const caps = this.ctx.agent.getCapabilities();
    console.log(`    Actions: ${caps.actions.length}`);
    console.log(`    Tools:   ${caps.tools.length}`);
    console.log('');
  }

  private async printWorkspaceMemory(args: string): Promise<void> {
    if (!this.ctx) return;

    const parts = args.trim().split(/\s+/).filter(Boolean);
    const scope = parts[0];
    const key = parts[1];

    if (scope === 'status' || !scope) {
      await this.printMemoryStatus();
      return;
    }

    if (scope === 'public') {
      await this.printPublicMemory(key);
      return;
    }

    if (scope === 'agent') {
      await this.printAgentMemory(key ?? 'roy');
      return;
    }

    if (scope === 'team') {
      if (!key) {
        console.log('\n  Usage: /memory team <team-key>\n');
        return;
      }
      const memory = await runtime.readTeamMemoryDoc(key, 'memory');
      console.log('\n  ' + this.bold(`Team Memory: ${key ?? ''}`));
      console.log(memory.trim() || this.dim('No team memory content'));
      console.log('');
      return;
    }

    if (scope === 'proposals') {
      await this.printMemoryProposals();
      return;
    }

    if (scope === 'show') {
      await this.printMemoryProposal(key);
      return;
    }

    if (scope === 'signals') {
      await this.printMemorySignals();
      return;
    }

    if (scope === 'summarize') {
      const summary = await runtime.summarizeMemoryUpdates();
      console.log('\n  ' + this.bold('Memory Summary'));
      console.log(`    Created this run: ${this.cyan(String(summary.createdThisRun))} proposal(s)`);
      console.log(`    Skipped duplicates: ${this.cyan(String(summary.skippedDuplicates))}`);
      console.log(`    Pending proposals: ${this.cyan(String(summary.pendingProposals))}`);
      console.log(`    Already committed: ${this.cyan(String(summary.alreadyCommitted))}`);
      console.log(`    Updated pending proposals: ${this.cyan(String(summary.updatedPendingProposals))}`);
      console.log('');
      return;
    }

    if (scope === 'accept') {
      if (!key) {
        console.log('\n  Usage: /memory accept <proposalId>\n');
        return;
      }
      const record = await runtime.acceptMemoryProposal(key);
      console.log(record
        ? `\n  ${this.green('Committed memory proposal')} ${key}\n`
        : `\n  ${this.yellow('No pending proposal found:')} ${key}\n`);
      return;
    }

    if (scope === 'reject') {
      if (!key) {
        console.log('\n  Usage: /memory reject <proposalId>\n');
        return;
      }
      const rejected = await runtime.rejectMemoryProposal(key);
      console.log(rejected
        ? `\n  ${this.green('Rejected memory proposal')} ${key}\n`
        : `\n  ${this.yellow('No pending proposal found:')} ${key}\n`);
      return;
    }

    if (scope === 'updates') {
      const updates = await runtime.listMemoryUpdates();
      console.log('\n  ' + this.bold('Memory Updates'));
      if (updates.length === 0) {
        console.log('    ' + this.dim('No committed memory updates.'));
      } else {
        for (const update of updates.slice(-20)) {
          console.log(`    - ${this.cyan(update.id)} ${this.dim(path.relative(process.cwd(), update.targetPath))} ${update.section ?? ''}`);
        }
      }
      console.log('');
      return;
    }

    if (scope === 'mode') {
      if (!key) {
        console.log('\n  Memory mode: ' + this.cyan(await runtime.getMemoryMode()) + '\n');
        return;
      }
      if (!['suggest', 'auto-safe', 'off'].includes(key)) {
        console.log('\n  Usage: /memory mode <suggest|auto-safe|off>\n');
        return;
      }
      await runtime.setMemoryMode(key as 'suggest' | 'auto-safe' | 'off');
      console.log('\n  Memory mode: ' + this.cyan(key) + '\n');
      return;
    }

    if (scope === 'auto') {
      if (key === 'on') {
        await runtime.setMemoryMode('suggest');
        console.log('\n  Auto-propose: ' + this.green('enabled') + '\n');
      } else if (key === 'off') {
        await runtime.setMemoryMode('off');
        console.log('\n  Auto-propose: ' + this.yellow('disabled') + '\n');
      } else {
        const state = await runtime.getMemoryAutoState();
        console.log('\n  Auto-propose: ' + (state.enabled ? this.green('enabled') : this.yellow('disabled')) + '\n');
      }
      return;
    }

    if (['project', 'context', 'decisions', 'constraints', 'glossary', 'user'].includes(scope)) {
      await this.printPublicMemory(scope);
      return;
    }

    console.log('\n  Usage: /memory [status|public|agent|proposals|show|signals|summarize|accept|reject|updates|mode]\n');
  }

  private async printMemoryStatus(): Promise<void> {
    const state = await runtime.getMemoryState();
    const mode = await runtime.getMemoryMode();
    const auto = await runtime.getMemoryAutoState();
    const proposals = await runtime.listMemoryProposals();
    console.log('\n  ' + this.bold('Workspace Memory'));
    console.log(`    Path:      ${this.cyan(path.relative(process.cwd(), state.rootPath) || state.rootPath)}`);
    console.log(`    Mode:      ${this.cyan(mode)}`);
    console.log(`    Auto-propose: ${auto.enabled ? this.green('enabled') : this.yellow('disabled')}`);
    if (auto.lastAutoPropose) {
      console.log('    Last auto-propose:');
      console.log(`      source:     ${auto.lastAutoPropose.source}`);
      console.log(`      session:    ${auto.lastAutoPropose.sessionId}`);
      console.log(`      created:    ${auto.lastAutoPropose.createdThisRun}`);
      console.log(`      skipped:    ${auto.lastAutoPropose.skippedDuplicates}`);
      console.log(`      updated:    ${auto.lastAutoPropose.updatedPendingProposals}`);
      console.log(`      pending:    ${auto.lastAutoPropose.pendingProposals}`);
      console.log(`      committed:  ${auto.lastAutoPropose.alreadyCommitted}`);
      if (auto.lastAutoPropose.reason) console.log(`      reason:     ${auto.lastAutoPropose.reason}`);
    }
    console.log(`    Public:    ${state.publicMemoryDocs.length} docs`);
    console.log(`    Agents:    ${state.agentMemories.length} memories`);
    console.log(`    Proposals: ${proposals.length} pending`);
    console.log(`    Traces:    ${state.traces}`);
    console.log(`    Queue:     ${path.relative(process.cwd(), state.queuePath) || state.queuePath}`);
    console.log(`    Patterns:  ${state.patterns.agents} agents, ${state.patterns.teams} teams, ${state.patterns.delegations} delegations`);

    if (state.publicMemoryDocs.length > 0) {
      console.log('\n  ' + this.bold('Public Memory:'));
      for (const doc of state.publicMemoryDocs) {
        console.log(`    - ${doc.name.padEnd(16)} ${doc.size} bytes`);
      }
    }

    if (state.agentMemories.length > 0) {
      console.log('\n  ' + this.bold('Agent Memories:'));
      for (const agent of state.agentMemories) {
        console.log(`    - ${this.cyan(agent.id)} ${this.dim(`${agent.docs.length} docs`)}`);
      }
    }

    console.log('\n  ' + this.bold('Agent Session Memory'));
    this.printMemory();
  }

  private async printPublicMemory(doc?: string): Promise<void> {
    if (!doc) {
      const state = await runtime.getMemoryState();
      console.log('\n  ' + this.bold('Public Memory'));
      for (const item of state.publicMemoryDocs) {
        console.log(`    - ${this.cyan(item.name.replace(/\.md$/, ''))} ${this.dim(`${item.size} bytes`)}`);
      }
      console.log('');
      return;
    }

    const content = await runtime.readPublicMemoryDoc(doc);
    console.log('\n  ' + this.bold(`Public Memory: ${doc}`));
    console.log(this.dim('  ' + '-'.repeat(58)));
    console.log(content.trim() ? content.trim() : this.dim('No content'));
    console.log('');
  }

  private async printAgentMemory(agentKey: string): Promise<void> {
    const memory = await runtime.readAgentMemoryDoc(agentKey, 'memory');
    const context = await runtime.readAgentMemoryDoc(agentKey, 'context');
    const prompt = await runtime.readAgentMemoryDoc(agentKey, 'prompt');

    console.log('\n  ' + this.bold(`Agent Memory: ${agentKey}`));
    console.log(this.dim('  ' + '-'.repeat(58)));
    console.log('\n  ' + this.bold('prompt.md'));
    console.log(prompt.trim() ? prompt.trim() : this.dim('No content'));
    console.log('\n  ' + this.bold('context.md'));
    console.log(context.trim() ? context.trim() : this.dim('No content'));
    console.log('\n  ' + this.bold('memory.md'));
    console.log(memory.trim() ? memory.trim() : this.dim('No content'));
    console.log('');
  }

  private async printMemoryProposals(): Promise<void> {
    const proposals = await runtime.listMemoryProposals();
    console.log('\n  ' + this.bold('Memory Proposals'));
    if (proposals.length === 0) {
      console.log('    ' + this.dim('No pending proposals.'));
      console.log('');
      return;
    }

    proposals.forEach((proposal, index) => {
      console.log(`\n  ${index + 1}. ${this.cyan(proposal.id)}`);
      console.log(`     target: ${path.relative(process.cwd(), proposal.target.path) || proposal.target.path}`);
      console.log(`     section: ${proposal.target.section ?? '-'}`);
      console.log(`     risk: ${proposal.risk}`);
      console.log(`     confidence: ${proposal.confidence}`);
      console.log(`     reason: ${proposal.reason}`);
    });
    console.log('');
  }

  private async printMemoryProposal(id?: string): Promise<void> {
    if (!id) {
      console.log('\n  Usage: /memory show <proposalId>\n');
      return;
    }
    const proposal = await runtime.getMemoryProposal(id);
    console.log('\n  ' + this.bold(`Memory Proposal: ${id}`));
    if (!proposal) {
      console.log('    ' + this.dim('No proposal found.'));
      console.log('');
      return;
    }
    console.log(`    status:     ${proposal.status}`);
    console.log(`    target:     ${path.relative(process.cwd(), proposal.target.path) || proposal.target.path}`);
    console.log(`    section:    ${proposal.target.section ?? '-'}`);
    console.log(`    operation:  ${proposal.operation}`);
    console.log(`    risk:       ${proposal.risk}`);
    console.log(`    confidence: ${proposal.confidence}`);
    console.log(`    reason:     ${proposal.reason}`);
    console.log('\n  ' + this.bold('Content:'));
    console.log(String(proposal.content).trim() ? String(proposal.content).trim() : this.dim('No content'));
    console.log('');
  }

  private async printMemorySignals(): Promise<void> {
    const signals = await runtime.collectMemorySignals();
    console.log('\n  ' + this.bold('Memory Signals'));
    console.log('  ' + this.bold('Source:'));
    console.log(`    session: ${this.cyan(signals.source.sessionId)}`);
    console.log(`    path:    ${this.dim(path.relative(process.cwd(), signals.source.sessionPath) || signals.source.sessionPath)}`);
    console.log(`    trace:   ${signals.source.traceName ?? 'none'}`);
    console.log('\n  ' + this.bold('Detected session records:'));
    console.log(`    user commands:         ${signals.counts.userCommands}`);
    console.log(`    agent results:         ${signals.counts.agentResults}`);
    console.log(`    root final responses:  ${signals.counts.rootFinalResponses}`);
    console.log(`    grounded results:      ${signals.counts.groundedAgentResults}`);
    console.log(`    tool calls:            ${signals.toolCalls.join(', ') || 'none'}`);
    console.log('\n  ' + this.bold('Agents:'));
    if (signals.agents.length === 0) {
      console.log('    ' + this.dim('No agent result records.'));
    } else {
      for (const agent of signals.agents) {
        console.log(`    - ${this.cyan(agent.agentId)} archetype=${agent.archetype} parent=${agent.parentId ?? '-'} grounded=${agent.grounded} outputGrounded=${agent.outputGrounded} tools=${agent.toolCalls.join(',') || 'none'}`);
      }
    }
    console.log('\n  ' + this.bold('Candidate memory signals:'));
    if (signals.candidateSignals.length === 0) {
      console.log('    ' + this.dim('No candidate signals.'));
    } else {
      for (const signal of signals.candidateSignals) {
        console.log(`    - ${signal}`);
      }
    }
    console.log('');
  }

  private printSession(): void {
    if (!this.ctx) return;

    const sessions = this.ctx.manager.listSessions();
    const agents = this.ctx.manager.listAgents();
    const messages = this.ctx.agent.getRecentMessages(100);

    console.log('\n  ' + this.bold('Session Information:'));
    console.log(`    Current Session: ${this.cyan(this.sessionId)}`);
    console.log(`    Total Sessions: ${sessions.length}`);
    console.log(`    Active Agents: ${agents.length}`);
    console.log(`    Memory Entries: ${messages.length}`);
    console.log('');
  }

  private printSystemInfo(): void {
    console.log('\n  ' + this.bold('System Information:'));
    console.log(`    Roy Version: ${this.cyan('0.1.0')}`);
    console.log(`    Node.js: ${process.version}`);
    console.log(`    Platform: ${process.platform}`);
    console.log(`    Config: ${path.resolve(process.cwd(), 'roy.config.yaml')}`);
    console.log('');
  }

  private printFSMInfo(): void {
    if (!this.ctx) return;

    const fsm = this.ctx.fsm;
    const ctx = fsm.getContext();

    console.log('\n  ' + this.bold('FSM State:'));
    console.log(`    Current: ${this.cyan(fsm.getStateName())}`);
    console.log(`    Budget: ${ctx.budget === null ? 'unlimited' : ctx.budget}`);
    console.log(`    Cost: ${ctx.cost}`);
    console.log(`    Uncertainty: ${ctx.uncertainty.toFixed(2)}`);
    console.log(`    Conflict: ${ctx.conflict.toFixed(2)}`);
    console.log(`    Evidence: ${ctx.evidence.toFixed(2)}`);

    if (ctx.trace.length > 0) {
      console.log('\n  ' + this.bold('Trace:'));
      for (const entry of ctx.trace.slice(-5)) {
        console.log(`    ${this.dim(entry.substring(0, 70) + (entry.length > 70 ? '...' : ''))}`);
      }
    }
    console.log('');
  }

  private async handleContext(parts: string[]): Promise<void> {
    if (parts[1] === 'render') {
      const agentKey = parts[2] ?? 'roy';
      const task = this.optionValue(parts, '--task') ?? this.trailingTask(parts, 3);
      const rendered = await runtime.renderAgentContext({
        agentKey,
        role: agentKey === 'roy' ? 'root' : 'subagent',
        task,
      });
      console.log('\n  ' + this.bold(`Context Render: ${agentKey}`));
      console.log('  ' + this.bold('Context Sources:'));
      console.log('    ' + this.dim(JSON.stringify(rendered.sources, null, 2).replace(/\n/g, '\n    ')));
      console.log('  ' + this.bold('Estimated Tokens:'));
      for (const [key, value] of Object.entries(rendered.tokenUsage)) {
        console.log(`    ${key}: ${this.cyan(String(value))}`);
      }
      console.log('');
      return;
    }
    this.printContext();
  }

  private printContext(): void {
    if (!this.ctx) return;

    console.log('\n  ' + this.bold('Conversation Context:'));
    const messages = this.ctx.agent.getRecentMessages(10);

    if (messages.length === 0) {
      console.log('    ' + this.dim('No messages in agent memory'));
    } else {
      for (const msg of messages) {
        const role = msg.role === 'system' ? this.yellow('[system]') : this.cyan('[' + msg.role + ']');
        const preview = msg.content.substring(0, 80) + (msg.content.length > 80 ? '...' : '');
        console.log(`    ${role} ${this.dim(preview)}`);
      }
      if (messages.length >= 10) {
        console.log(`    ${this.dim('... showing last 10 of ${messages.length} messages')}`);
      }
    }
    console.log('');
  }

  private async printAgentPrompt(agentKey: string): Promise<void> {
    const prompt = await runtime.readAgentMemoryDoc(agentKey, 'prompt');
    console.log('\n  ' + this.bold(`Prompt: ${agentKey}`));
    console.log(this.dim('  ' + '-'.repeat(58)));
    console.log(prompt.trim() ? prompt.trim() : this.dim('No prompt.md content'));
    console.log('');
  }

  private async renderPrompt(parts: string[]): Promise<void> {
    const agentKey = parts[2] ?? 'roy';
    const name = this.optionValue(parts, '--name');
    const role = this.optionValue(parts, '--role');
    const parentId = this.optionValue(parts, '--parent');
    const task = this.optionValue(parts, '--task') ?? this.trailingTask(parts, 3);
    const rendered = await runtime.renderAgentPrompt({
      agentKey,
      name,
      role,
      parentId,
      task,
      archetype: ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'].includes(agentKey)
        ? agentKey as SubAgentArchetype
        : undefined,
    });
    console.log('\n  ' + this.bold(`Rendered Prompt: ${agentKey}`));
    console.log(`    estimated tokens: ${this.cyan(String(rendered.estimatedTokens))}`);
    console.log('    sources: ' + this.dim(JSON.stringify(rendered.sources)));
    console.log(this.dim('  ' + '-'.repeat(58)));
    console.log(rendered.prompt.trim());
    console.log('');
  }

  private completer(line: string): [string[], string] {
    const commands = [
      '/help', '/h', '/clear', '/cls', '/reset', '/agents', '/spawn', '/run', '/teams', '/team', '/tom', '/exit', '/quit', '/q',
      '/api', '/status', '/skills', '/actions', '/tools', '/memory', '/session',
      '/system', '/fsm', '/budget', '/events', '/queue', '/cache', '/messages', '/traces', '/config', '/prompt', '/context', '/conversation', '/verbose', '/color'
    ];

    if (line.startsWith('/')) {
      const hits = commands.filter(c => c.startsWith(line.toLowerCase()));
      return [hits.length ? hits : commands, line];
    }

    return [[], line];
  }

  private parseCommand(input: string): string[] {
    const matches = input.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
    return matches.map(part => {
      if ((part.startsWith('"') && part.endsWith('"')) || (part.startsWith("'") && part.endsWith("'"))) {
        return part.slice(1, -1);
      }
      return part;
    });
  }

  private async spawnAgent(parts: string[]): Promise<void> {
    const archetype = parts[1] as SubAgentArchetype;
    const quiet = parts.includes('--quiet');
    const parentId = this.optionValue(parts, '--parent');
    const name = this.optionValue(parts, '--name');
    const customRole = this.optionValue(parts, '--role');
    const customStyle = this.optionValue(parts, '--style');
    const tools = this.optionList(parts, '--tools');
    const skills = this.optionList(parts, '--skills');
    const task = this.trailingTask(parts, 2);
    const allowed = ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'];

    if (parts.includes('--help') || !allowed.includes(archetype) || !task) {
      this.printSpawnUsage();
      return;
    }

    try {
      const result = await runtime.handleSpawnCommand({
        archetype,
        task,
        parentId,
        name,
        customRole,
        customStyle,
        tools,
        skills,
        requireRootSynthesis: true,
        showSubagentOutput: !quiet,
      });
      console.log(`\n  ${this.dim(`message chain: ${result.correlationId}`)}`);
      const events = runtime.getEvents().filter(event => event.type === 'cache.hit' && event.data?.correlationId === result.correlationId);
      for (const event of events.slice(-4)) {
        const patternId = event.data?.patternId;
        if (typeof patternId === 'string') {
          console.log(`  ${this.green('[event]')} cache.hit ${patternId}`);
        }
      }
      console.log(`  ${this.dim(`agent creation: mode=${result.creationUsage.mode}, definition=${result.creationUsage.definitionTokens} tokens, rendered=${result.creationUsage.renderedPromptTokens} tokens (${result.creationUsage.renderedPromptChars} chars), cache hits=${result.creationUsage.cacheHits.length}`)}`);
      console.log(`  ${this.dim(`node: ${result.node.nodeId}, definition=${result.node.definitionFingerprint.slice(0, 12)}, invocation=${result.node.invocationFingerprint.slice(0, 12)}`)}`);
      console.log(`  ${this.yellow('roy[root] delegating...')}`);
      console.log(`  ├─ ${this.yellow(`${result.agent.name}[subagent] thinking...`)}`);
      if (result.subagentResult.toolCalls.length > 0) {
        for (const call of result.subagentResult.toolCalls) {
          const toolPath = typeof call.params.path === 'string'
            ? path.relative(process.cwd(), call.params.path) || '.'
            : '';
          console.log(`  │  ├─ ${this.dim(`tool: ${call.toolName} ${toolPath}`.trim())}`);
        }
      }
      console.log(`  │  └─ ${this.green(`completed, ${result.subagentResult.usage.totalTokens} tokens`)}`);
      if (!quiet && result.subagentResult.result) {
        console.log(`  │     ${this.dim(`${result.agent.name}[subagent] report:`)}`);
        for (const line of result.subagentResult.result.split('\n').slice(0, 12)) {
          console.log(`  │     ${this.dim(line)}`);
        }
      }
      if (result.subagentResult.warnings.length > 0) {
        for (const warning of result.subagentResult.warnings) {
          console.log(`  ${this.yellow('[warning]')} ${warning}`);
        }
      }
      console.log(`  └─ ${this.yellow('roy[root] synthesizing...')}\n`);
      console.log('  ' + this.green('roy[root] > ') + result.finalResponse + '\n');
    } catch (error) {
      console.log('\n  ' + this.red('Spawn error:') + ' ' + (error instanceof Error ? error.message : String(error)) + '\n');
    }
  }

  private async printCache(parts: string[]): Promise<void> {
    const scope = parts[1] ?? 'agents';
    if (scope === 'evolution') {
      const records = await runtime.getEvolutionHistory(20);
      console.log('\n  ' + this.bold('Cache: evolution'));
      if (records.length === 0) console.log('    ' + this.dim('No evolution runs recorded.'));
      for (const record of records) {
        console.log(`    - ${this.cyan(String(record.correlationId ?? 'run'))} parent=${record.parentId ?? '-'} selected=${record.selected ?? 'none'}`);
      }
      console.log('');
      return;
    }
    const kind = scope === 'delegations' || scope === 'teams' ? scope : 'agents';
    const patterns = await runtime.getCachePatterns(kind);
    console.log('\n  ' + this.bold(`Cache: ${kind}`));
    if (patterns.length === 0) {
      console.log('    ' + this.dim('No cached patterns.'));
      console.log('');
      return;
    }
    for (const pattern of patterns) {
      const usage = typeof pattern.usage === 'object' && pattern.usage !== null ? pattern.usage as Record<string, unknown> : {};
      console.log(`    - ${this.cyan(String(pattern.id ?? pattern.key ?? 'pattern'))}`);
      console.log(`      usage.count: ${usage.count ?? 0}`);
      console.log(`      lastUsedAt:  ${usage.lastUsedAt ?? '-'}`);
      if (pattern.promptPath) console.log(`      promptPath:  ${pattern.promptPath}`);
      if (pattern.memoryPath) console.log(`      memoryPath:  ${pattern.memoryPath}`);
      if (Array.isArray(pattern.tools)) console.log(`      tools:       ${pattern.tools.join(', ') || 'none'}`);
      if (Array.isArray(pattern.skills)) console.log(`      skills:      ${pattern.skills.join(', ') || 'none'}`);
      if (typeof pattern.spawnPolicy === 'object' && pattern.spawnPolicy !== null) {
        const policy = pattern.spawnPolicy as Record<string, unknown>;
        console.log(`      spawnPolicy: maxChildren=${policy.maxChildren ?? '-'}, maxDepth=${policy.maxDepth ?? '-'}, budgetAware=${policy.budgetAware ?? '-'}`);
      }
      if (usage.definitionTokensSaved !== undefined) console.log(`      savedTokens: ${usage.definitionTokensSaved}`);
      if (usage.lastRenderedPromptTokens !== undefined) console.log(`      rendered:    ${usage.lastRenderedPromptTokens} tokens`);
    }
    console.log('');
  }

  private printSpawnUsage(): void {
    console.log(`
  Usage:
    /spawn <researcher|critic|planner|coder|summarizer|tester> "task"
    /spawn custom --name <agentName> [--role <role>] [--style <style>] "task"
    /spawn custom --name <agentName> [--tools fs.read,fs.list] [--skills use_tool_when_needed] "task"
    /spawn <archetype> --parent <agentId> "task"

  Examples:
    /spawn researcher "Inspect the project structure"
    /spawn critic --parent agent_researcher_001 "Review Researcher-1's report"
    /spawn custom --name "Singer-1" --role "performer" "Write a short original song"

  Notes:
    - The first argument is an agent archetype, not an agent name.
    - Custom names must be passed with --name.
    - --tools and --skills are parent-approved bindings for the child agent.
    - A cache hit reuses agent pattern/config, but still creates a new runtime agent instance.
`);
  }

  private optionValue(parts: string[], option: string): string | undefined {
    const index = parts.indexOf(option);
    return index >= 0 ? parts[index + 1] : undefined;
  }

  private optionList(parts: string[], option: string): string[] | undefined {
    const value = this.optionValue(parts, option);
    if (!value) return undefined;
    const items = value.split(',').map(item => item.trim()).filter(Boolean);
    return items.length > 0 ? items : undefined;
  }

  private trailingTask(parts: string[], startIndex: number): string {
    const optionsWithValues = new Set(['--parent', '--name', '--role', '--style', '--task', '--tools', '--skills']);
    const taskParts: string[] = [];
    for (let index = startIndex; index < parts.length; index += 1) {
      const part = parts[index];
      if (part === '--quiet' || part === '--help') continue;
      if (optionsWithValues.has(part)) {
        index += 1;
        continue;
      }
      taskParts.push(part);
    }
    return taskParts.join(' ');
  }

  private async printMessages(parts: string[]): Promise<void> {
    const correlationIndex = parts.indexOf('--correlation');
    const correlationId = correlationIndex >= 0 ? parts[correlationIndex + 1] : undefined;
    const messages = await runtime.getMessages({ correlationId, limit: correlationId ? undefined : 20 });

    console.log('\n  ' + this.bold(correlationId ? `Messages: ${correlationId}` : 'Recent Messages'));
    if (messages.length === 0) {
      console.log('    ' + this.dim('No messages'));
    } else {
      for (const message of messages) {
        console.log(`    ${message.from} -> ${message.to}: ${this.cyan(message.kind)} ${this.dim(message.status)}`);
      }
    }
    console.log('');
  }

  private async printConversation(parts: string[]): Promise<void> {
    if (parts[1] === 'sessions') {
      const sessions = await runtime.listConversationSessions();
      console.log('\n  ' + this.bold('Conversation Sessions'));
      if (sessions.length === 0) {
        console.log('    ' + this.dim('No persisted sessions'));
      } else {
        for (const session of sessions) {
          console.log(`    ${this.cyan(session.sessionId)} ${this.dim(`${session.entries} entries`)} ${this.dim(new Date(session.updatedAt).toISOString())}`);
          console.log(`      ${path.relative(process.cwd(), session.path) || session.path}`);
        }
      }
      console.log('');
      return;
    }

    if (parts[1] === 'import') {
      const filePath = parts.slice(2).join(' ');
      if (!filePath) {
        console.log('\n  Usage: /conversation import <path-to-json-or-jsonl>\n');
        return;
      }
      try {
        const result = await runtime.importConversation(path.resolve(process.cwd(), filePath));
        console.log(`\n  ${this.green('Imported')} ${result.imported} conversation entries into ${this.cyan(path.relative(process.cwd(), result.path) || result.path)}\n`);
      } catch (error) {
        console.log('\n  ' + this.red('Import error:') + ' ' + (error instanceof Error ? error.message : String(error)) + '\n');
      }
      return;
    }

    const sessionFlagIndex = parts.indexOf('--session');
    const sessionId = sessionFlagIndex >= 0 ? parts[sessionFlagIndex + 1] : undefined;
    const limitArg = parts.find((part, index) => index > 0 && part !== '--session' && parts[index - 1] !== '--session');
    const limit = Number(limitArg ?? 20);
    const entries = await runtime.getConversation(sessionId, Number.isFinite(limit) && limit > 0 ? limit : 20);

    console.log('\n  ' + this.bold(sessionId ? `Persisted Conversation: ${sessionId}` : 'Persisted Conversation'));
    if (entries.length === 0) {
      console.log('    ' + this.dim('No persisted conversation entries'));
    } else {
      for (const entry of entries) {
        const when = new Date(entry.timestamp).toISOString();
        console.log(`    ${this.dim(when)} ${this.cyan(entry.speaker)} [${entry.role}]`);
        console.log(`      ${entry.content.substring(0, 140)}${entry.content.length > 140 ? '...' : ''}`);
      }
    }
    console.log('');
  }

  private async runAgent(parts: string[]): Promise<void> {
    const agentId = parts[1];
    const task = parts.slice(2).join(' ');

    if (!agentId || !task) {
      console.log('\n  Usage: /run <agent-id> "task"\n');
      return;
    }

    try {
      console.log(`\n  ${this.yellow(`${agentId} thinking...`)}\n`);
      const result = await runtime.runAgent(agentId, task);
      if (result.result) {
        console.log('  ' + this.green(`${result.agent.name}[subagent] > `) + result.result);
      }
      console.log(`\n  ${this.green('[event]')} agent.run.completed: ${agentId}, tokens=${result.usage.totalTokens}\n`);
    } catch (error) {
      console.log('\n  ' + this.red('Run error:') + ' ' + (error instanceof Error ? error.message : String(error)) + '\n');
    }
  }

  private async processMessage(userInput: string): Promise<void> {
    if (!this.ctx) {
      console.log('\n  ' + this.red('Not ready.') + '\n');
      return;
    }

    console.log('\n  ' + this.yellow('roy[root] thinking...') + '\n');

    try {
      const result = await runtime.handleUserTurn(userInput);
      await this.printRootTurnResult(result);

      // Show updated FSM state
      if (this.verboseMode) {
        const fsm = this.ctx.fsm;
        console.log('\n  ' + this.dim(`[FSM: ${fsm.getStateName()}, Cost: ${fsm.getContext().cost}]`));
      }
    } catch (error) {
      console.log('\n  ' + this.red('Error:') + ' ' + (error instanceof Error ? error.message : String(error)) + '\n');
      logger.error('Process message error:', error);
    }
  }

  private async printRootTurnResult(result: Awaited<ReturnType<typeof runtime.handleUserTurn>>): Promise<void> {
    if (result.decision.action === 'spawn_subagents') {
      console.log(`  ${this.green('roy[root] spawned')} ${result.subagents.length} subagent${result.subagents.length === 1 ? '' : 's'}: ${result.subagents.map(item => item.agent.name).join(', ')}`);
      if (this.verboseMode) {
        console.log(`  ${this.dim(`decision: ${result.decision.reason}`)}`);
      }
      for (let index = 0; index < result.subagents.length; index += 1) {
        const item = result.subagents[index];
        const branch = index === result.subagents.length - 1 ? '└─' : '├─';
        console.log(`  ${branch} ${this.yellow(`${item.agent.name}[subagent] completed`)}, ${item.subagentResult.usage.totalTokens} tokens`);
        if (this.verboseMode) {
          console.log(`  │  ${this.dim(`node: ${item.node.nodeId} definition=${item.node.definitionFingerprint.slice(0, 12)}`)}`);
          for (const call of item.subagentResult.toolCalls) {
            const toolPath = typeof call.params.path === 'string'
              ? path.relative(process.cwd(), call.params.path) || '.'
              : '';
            console.log(`  │  ${this.dim(`tool: ${call.toolName} ${toolPath}`.trim())}`);
          }
          for (const warning of item.subagentResult.warnings) {
            console.log(`  │  ${this.yellow('[warning]')} ${warning}`);
          }
        }
      }
      console.log(`  ${this.yellow('roy[root] synthesizing...')}\n`);
    } else if (this.verboseMode) {
      console.log(`  ${this.dim(`decision: solve_directly - ${result.decision.reason}`)}\n`);
    }

    console.log('  ' + this.green('roy[root] > ') + result.finalResponse + '\n');
  }

  private async printAgentOutput(): Promise<string> {
    if (!this.ctx) return '';

    const session = this.ctx.manager.getSession(this.sessionId);
    if (!session) return '';

    let printed = false;
    const chunks: string[] = [];
    while (!session.messageQueue.isEmpty('env')) {
      const message = await session.messageQueue.receive('env');
      if (!message) break;

      const content = String(message.content);
      if (content.length > 0) {
        chunks.push(content);
        if (!printed) {
          process.stdout.write('  ' + this.green(`${this.ctx.agent.name.toLowerCase()}[root] > `));
        }
        process.stdout.write(content);
        printed = true;
      }
    }

    if (printed) {
      process.stdout.write('\n');
    }
    return chunks.join('');
  }
}

// CLI entry point
async function main(): Promise<void> {
  const roy = new Roy();
  await roy.launch();
}

main().catch(console.error);
