// Roy CLI - Terminal Interface for the Agent System

import * as readline from 'readline';
import * as path from 'path';
import { bootstrap, cleanup, type BootstrapContext } from '../bootstrap.js';
import { runtime } from '../core/runtime/Runtime.js';
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
  private rl: readline.Interface;
  private autoColor = true;
  private verboseMode = false;
  private sessionId = 'cli-session';

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      completer: this.completer.bind(this),
    });
  }

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

  private async launch(): Promise<void> {
    this.printBanner();

    try {
      this.ctx = await bootstrap({
        agentName: 'Roy',
        agentGoal: 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
        sessionId: this.sessionId,
        fsmEnabled: true,
      });

      logger.info('CLI Bootstrap complete');
      this.printReady();
      this.startChat();
    } catch (error) {
      console.log('\n  ' + this.red('[ERROR]') + ' Failed to initialize Roy');
      console.log('  ' + this.dim(String(error)));
      logger.error('Bootstrap failed:', error);
      process.exit(1);
    }
  }

  private printBanner(): void {
    console.log(this.green(BANNER));
    console.log(this.dim('='.repeat(60)));
  }

  private printReady(): void {
    if (!this.ctx) return;

    const state = runtime.getState();
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
      this.rl.question(question, async (answer) => {
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
      this.rl.question('\n  ' + message + ' ', () => {
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

  private startChat(): void {
    const prompt = () => {
      this.rl.question(this.cyan('\nyou') + ' > ', async (input) => {
        const trimmed = input.trim();

        if (!trimmed) {
          prompt();
          return;
        }

        if (trimmed.startsWith('/')) {
          const shouldContinue = await this.handleCommand(trimmed);
          if (shouldContinue !== false) {
            prompt();
          }
          return;
        }

        await this.processMessage(trimmed);
        prompt();
      });
    };

    prompt();

    this.rl.on('close', async () => {
      console.log('\n\n' + this.yellow('Goodbye! Roy shutting down...') + '\n');
      if (this.ctx) {
        await cleanup(this.ctx);
      }
      process.exit(0);
    });
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
        this.printAgents(parts.includes('--tree'));
        break;

      case '/spawn':
        await this.spawnAgent(parts);
        break;

      case '/run':
        await this.runAgent(parts);
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
        this.printEvents();
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
        this.printTools();
        break;

      case '/memory':
        this.printMemory();
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
        if (args && this.ctx) {
          this.ctx.agent.addToMemory('meta', `System prompt: ${args}`);
          console.log('\n  ' + this.green('System prompt added to agent memory') + '\n');
        } else {
          console.log('\n  Usage: /prompt <system instructions>' + '\n');
        }
        break;

      case '/context':
        this.printContext();
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

    ${this.bold('System Information')}
      /help, /h           Show this help message
      /status             Show connection status with FSM state
      /system             Show system information
      /fsm                Show FSM state and trace
      /budget             Show token budget and usage
      /budget set <n>     Set token budget
      /budget unlimited   Remove token budget limit
      /events             Show recent runtime events
      /verbose            Toggle verbose mode

    ${this.bold('Agent Management')}
      /agents             List available agents
      /agents --tree      Show agent parent-child tree
      /spawn <type> "task" Spawn and run a controlled subagent
      /run <agent-id> "task" Run an existing subagent
      /session            Show current session info
      /reset              Reset FSM to initial state

    ${this.bold('Capabilities')}
      /skills             List registered skills
      /actions            List available actions
      /tools              List available tools
      /memory             Show memory statistics

    ${this.bold('Configuration')}
      /api                Show API information
      /config             Show runtime configuration
      /prompt <text>      Add system prompt to agent memory
      /color              Toggle color output

    ${this.bold('Exit')}
      /exit, /quit, /q    Exit Roy
`);
  }

  private printAgents(tree = false): void {
    if (!this.ctx) return;

    if (tree) {
      console.log('\n  ' + this.bold('Agent Tree:'));
      this.printAgentTree(runtime.getAgentTree(), '    ', true);
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
      }
    }
    console.log('');
  }

  private printAgentTree(node: ReturnType<typeof runtime.getAgentTree>, prefix: string, isRoot = false): void {
    const agent = node.agent;
    const usage = agent.usage;
    const label = `${agent.name} [${agent.role}, ToM-${agent.identity.tomLevel}, ${agent.state}, ${usage.totalTokens} tokens]`;
    console.log(prefix + (isRoot ? '' : '└── ') + this.cyan(label));
    const childPrefix = prefix + (isRoot ? '' : '    ');
    for (let i = 0; i < node.children.length; i++) {
      this.printAgentTree(node.children[i], childPrefix, false);
    }
  }

  private printApiInfo(): void {
    console.log('\n  ' + this.bold('API Endpoints:'));
    console.log('    GET  /           - Server info');
    console.log('    GET  /health     - Health check');
    console.log('    GET  /v1/status  - Runtime status');
    console.log('    GET  /v1/agents  - Agent states');
    console.log('    GET  /v1/agents/tree - Agent tree');
    console.log('    POST /v1/agents  - Spawn subagent');
    console.log('    POST /v1/agents/:id/run - Run subagent');
    console.log('    GET  /v1/budget  - Token budget');
    console.log('    GET  /v1/events  - Runtime events');
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

    this.printBudget();
  }

  private printEvents(): void {
    const events = runtime.getEvents().slice(-10);
    console.log('\n  ' + this.bold('Recent Events'));
    if (events.length === 0) {
      console.log('    ' + this.dim('No events'));
    } else {
      for (const event of events) {
        console.log(`    - ${this.dim(new Date(event.timestamp).toISOString())} ${this.cyan(event.type)}${event.agentId ? ' ' + event.agentId : ''}`);
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

  private completer(line: string): [string[], string] {
    const commands = [
      '/help', '/h', '/clear', '/cls', '/reset', '/agents', '/spawn', '/run', '/exit', '/quit', '/q',
      '/api', '/status', '/skills', '/actions', '/tools', '/memory', '/session',
      '/system', '/fsm', '/budget', '/events', '/config', '/prompt', '/context', '/verbose', '/color'
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
    const archetype = parts[1] as any;
    const task = parts.slice(2).join(' ');
    const allowed = ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'];

    if (!allowed.includes(archetype) || !task) {
      console.log('\n  Usage: /spawn <researcher|critic|planner|coder|summarizer|tester|custom> "task"\n');
      return;
    }

    try {
      const agent = await runtime.spawnAgent({
        parentId: 'root',
        archetype,
        tomLevel: 2,
        description: task,
        task,
      });
      console.log(`\n  ${this.green('[event]')} agent.spawned: ${agent.identity.id} ${agent.name} parent=root`);
      console.log(`  ${this.yellow(`${agent.name}[subagent] thinking...`)}\n`);
      const result = await runtime.runAgent(agent.identity.id, task);
      if (result.result) {
        console.log('  ' + this.green(`${agent.name}[subagent] > `) + result.result);
      }
      console.log(`\n  ${this.green('roy[root] >')} Spawned ${agent.name} and completed the controlled subagent run.\n`);
    } catch (error) {
      console.log('\n  ' + this.red('Spawn error:') + ' ' + (error instanceof Error ? error.message : String(error)) + '\n');
    }
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
      // Use agent's step method which integrates FSM and memory
      this.ctx.agent.addToMemory('meta', 'user turn started');
      const usageBefore = this.ctx.agent.getUsage();
      runtime.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'thinking' } });
      await this.ctx.agent.step(userInput);
      const usageAfter = this.ctx.agent.getUsage();
      runtime.recordTurnUsage({
        llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
        promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
        completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
        totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
      });
      runtime.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle' } });
      await this.printAgentOutput();

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

  private async printAgentOutput(): Promise<void> {
    if (!this.ctx) return;

    const session = this.ctx.manager.getSession(this.sessionId);
    if (!session) return;

    let printed = false;
    while (!session.messageQueue.isEmpty('env')) {
      const message = await session.messageQueue.receive('env');
      if (!message) break;

      const content = String(message.content);
      if (content.length > 0) {
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
  }
}

// CLI entry point
async function main(): Promise<void> {
  const roy = new Roy();
  await (roy as any).launch();
}

main().catch(console.error);
