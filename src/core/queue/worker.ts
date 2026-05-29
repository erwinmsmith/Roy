import type { MessageKind, RuntimeMessage } from './types.js';

export interface MessageWorker {
  id: string;
  accepts(kind: MessageKind, message: RuntimeMessage): boolean;
  handle(message: RuntimeMessage): Promise<void>;
}

