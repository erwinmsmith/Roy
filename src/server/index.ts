// Server entry point with Socket.IO

import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { AgentManager } from '../core/manager/AgentManager.js';
import { ConversationalAgent } from '../core/agent/ConversationalAgent.js';
import { llmFactory } from '../core/llm/index.js';
import { logger } from '../core/utils/logger.js';

const PORT = parseInt(process.env.PORT || '3000', 10);
const LOG_LEVEL = (process.env.LOG_LEVEL || 'info') as 'debug' | 'info' | 'warn' | 'error';
logger.setLevel(LOG_LEVEL);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});

// Initialize AgentManager
const manager = new AgentManager();

// Create default LLM provider
let defaultLlm;
try {
  defaultLlm = llmFactory.getDefault();
  logger.info(`Default LLM provider: ${defaultLlm.name}`);
} catch (error) {
  logger.warn('No LLM providers configured - agent functionality limited');
}

// Create and add default agent
const defaultAgent = new ConversationalAgent({
  name: 'ConversationalAgent',
  goal: 'I am a conversational agent. Help users with their questions.',
  llm: defaultLlm,
});
manager.addAgent(defaultAgent);
manager.setInteractWithEnv('ConversationalAgent');

// Socket.IO connection handling
io.on('connection', (socket) => {
  const sid = socket.id;
  logger.info(`Client connected: ${sid}`);

  // Create session for this socket
  const session = manager.createSession(sid);

  socket.on('user_message', async (message: string) => {
    logger.info(`Received message from ${sid}: ${message.substring(0, 100)}...`);

    // Emit start signal
    socket.emit('bot_response_start', 'Bot: Hi, I am a conversational agent. How can I help you?');

    try {
      // Stream response
      for await (const chunk of manager.streamResponse(sid, message)) {
        socket.emit('bot_response_stream', chunk);
      }
    } catch (error) {
      logger.error(`Error processing message:`, error);
      socket.emit('bot_response_stream', `Error: ${error}`);
    }
  });

  socket.on('disconnect', async () => {
    logger.info(`Client disconnected: ${sid}`);
    await manager.closeSession(sid);
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    agents: manager.listAgents(),
    sessions: manager.listSessions(),
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Roy',
    description: 'Theory of Mind based autonomous derived MAS',
    version: '0.1.0',
    endpoints: {
      health: '/health',
      socket: 'Connect via Socket.IO',
    },
  });
});

// Start server
httpServer.listen(PORT, () => {
  logger.info(`Roy server running on port ${PORT}`);
  logger.info(`WebSocket endpoint: ws://localhost:${PORT}`);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('Shutting down...');
  await manager.cleanup();
  httpServer.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('SIGTERM', async () => {
  logger.info('Shutting down...');
  await manager.cleanup();
  httpServer.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

export { app, io, manager };