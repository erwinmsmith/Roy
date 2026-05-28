// Server entry point with Socket.IO - Unified via bootstrap

import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { bootstrap, cleanup, type BootstrapContext } from '../bootstrap.js';
import { logger } from '../core/utils/logger.js';
import type { QueueMessage } from '../core/message/MessageQueue.js';

// ASCII Banner - compatible with all terminals
const SERVER_BANNER = `
+========================================================+
|                                                        |
|     ██████╗  ██████╗ ██╗   ██╗                     |
|     ██╔══██╗██╔═══██╗╚██╗ ██╔╝                     |
|     ██████╔╝██║   ██║ ╚████╔╝                      |
|     ██╔══██╗██║   ██║  ╚██╔╝                       |
|     ██║  ██║╚██████╔╝   ██║                        |
|     ╚═╝  ╚═╝ ╚═════╝    ╚═╝                        |
|                                                        |
| Theory of Mind based Autonomous Agent System           |
|                                                        |
+========================================================+
`;

// Color utilities for server output
function green(text: string): string {
  return `\x1b[32m${text}\x1b[0m`;
}

function yellow(text: string): string {
  return `\x1b[33m${text}\x1b[0m`;
}

function red(text: string): string {
  return `\x1b[31m${text}\x1b[0m`;
}

function cyan(text: string): string {
  return `\x1b[36m${text}\x1b[0m`;
}

function bold(text: string): string {
  return `\x1b[1m${text}\x1b[0m`;
}

function dim(text: string): string {
  return `\x1b[2m${text}\x1b[0m`;
}

async function receiveWithTimeout(
  receive: () => Promise<QueueMessage | undefined>,
  timeoutMs: number
): Promise<QueueMessage | undefined> {
  return Promise.race([
    receive(),
    new Promise<undefined>(resolve => setTimeout(() => resolve(undefined), timeoutMs)),
  ]);
}

// Graceful port handling
function tryListen(server: ReturnType<typeof createServer>, port: number, retries = 3): Promise<number> {
  return new Promise((resolve, reject) => {
    const attempt = (currentPort: number, remainingRetries: number) => {
      server.listen(currentPort, () => {
        resolve(currentPort);
      }).on('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EADDRINUSE' && remainingRetries > 0) {
          logger.warn(`Port ${currentPort} in use, trying ${currentPort + 1}...`);
          attempt(currentPort + 1, remainingRetries - 1);
        } else {
          reject(err);
        }
      });
    };
    attempt(port, retries);
  });
}

async function main(): Promise<void> {
  // Bootstrap the system
  let ctx: BootstrapContext;

  try {
    ctx = await bootstrap({
      agentName: 'ServerAgent',
      agentGoal: 'I am a server-side conversational agent. Help users with their questions.',
      sessionId: 'server-main',
      fsmEnabled: true,
    });

    logger.info('Server Bootstrap complete');
  } catch (error) {
    logger.error('Bootstrap failed:', error);
    console.log(red('\n[ERROR] Failed to initialize server'));
    process.exit(1);
  }

  const PORT = ctx.config.server?.port ?? 3000;

  const app = express();
  const httpServer = createServer(app);

  // Socket.IO connection handling
  const io = new Server(httpServer, {
    cors: ctx.config.server?.cors ?? {
      origin: '*',
      methods: ['GET', 'POST'],
    },
  });

  io.on('connection', (socket) => {
    const sid = socket.id;
    logger.info(`Client connected: ${sid}`);

    // Create session for this socket
    ctx.manager.createSession(sid);

    socket.on('user_message', async (message: string) => {
      logger.info(`Received message from ${sid}: ${message.substring(0, 100)}...`);

      // Get the agent for this session
      const agent = ctx.agent;

      socket.emit('bot_response_start', 'Bot: Processing your request...');

      try {
        const session = ctx.manager.getSession(sid);
        if (!session) {
          throw new Error(`Session ${sid} not found`);
        }

        let stepDone = false;
        let stepErrorMessage: string | null = null;

        const stepPromise = agent.step(message)
          .catch(err => {
            stepErrorMessage = err instanceof Error ? err.message : String(err);
            logger.error('Agent step error:', err);
          })
          .finally(() => {
            stepDone = true;
          });

        while (!stepDone || !session.messageQueue.isEmpty('env')) {
          const queued = await receiveWithTimeout(
            () => session.messageQueue.receive('env'),
            stepDone ? 25 : 100
          );

          if (!queued) {
            continue;
          }

          socket.emit('bot_response_stream', String(queued.content));

          if (queued.metadata?.done === true) {
            break;
          }
        }

        await stepPromise;

        if (stepErrorMessage) {
          socket.emit('bot_response_stream', `Error: ${stepErrorMessage}`);
        }
      } catch (error) {
        logger.error(`Error processing message:`, error);
        socket.emit('bot_response_stream', `Error: ${error}`);
      }
    });

    socket.on('disconnect', async () => {
      logger.info(`Client disconnected: ${sid}`);
      await ctx.manager.closeSession(sid);
    });
  });

  // Health check endpoint
  app.get('/health', (req, res) => {
    const fsmInfo = ctx.agent.getFSMInfo();
    res.json({
      status: 'ok',
      name: 'Roy',
      version: '0.1.0',
      mode: 'api',
      agent: ctx.agent.name,
      fsm: fsmInfo ? {
        state: fsmInfo.state,
        cost: fsmInfo.cost,
        budget: fsmInfo.budget,
      } : null,
      agents: ctx.manager.listAgents(),
      sessions: ctx.manager.listSessions(),
      capabilities: ctx.capabilities,
      timestamp: new Date().toISOString(),
    });
  });

  // Root endpoint
  app.get('/', (req, res) => {
    res.json({
      name: 'Roy',
      description: 'Theory of Mind based autonomous derived MAS',
      version: '0.1.0',
      mode: 'api',
      endpoints: {
        health: '/health',
        websocket: 'Connect via Socket.IO',
      },
    });
  });

  // Start server with graceful port handling
  tryListen(httpServer, PORT)
    .then((actualPort) => {
      console.log(green(SERVER_BANNER));
      console.log(dim('='.repeat(60)));
      console.log(green('[OK]') + ' Server running on port ' + bold(String(actualPort)));
      console.log('  WebSocket: ' + cyan(`ws://localhost:${actualPort}`));
      console.log('  Health:    ' + cyan(`http://localhost:${actualPort}/health`));
      console.log(dim('='.repeat(60)));
      console.log('  Agent: ' + cyan(ctx.agent.name));
      console.log('  FSM: ' + cyan(ctx.fsm.getStateName()));
      console.log(dim('='.repeat(60)));
      console.log('');

      if (actualPort !== PORT) {
        console.log(yellow('[WARN] Port ' + PORT + ' was in use, started on ' + actualPort));
      }
    })
    .catch((err) => {
      logger.error('Failed to start server:', err);
      console.error(red('[ERROR] Failed to start server:') + ' ' + err.message);
      process.exit(1);
    });

  // Graceful shutdown
  const shutdown = async () => {
    console.log(yellow('\nShutting down...'));
    await cleanup(ctx);
    io.close();
    httpServer.close(() => {
      console.log(green('[OK] Server closed'));
      process.exit(0);
    });
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch(err => {
  logger.error('Unhandled error:', err);
  process.exit(1);
});
