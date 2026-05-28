// Logging Event types and interfaces

export type EventType = 'debug' | 'info' | 'warning' | 'error' | 'progress';

export interface EventContext {
  sessionId?: string;
  traceId?: string;
  spanId?: string;
  [key: string]: unknown;
}

export interface LogEvent {
  id: string;
  type: EventType;
  name?: string;
  namespace: string;
  message: string;
  context?: EventContext;
  data?: Record<string, unknown>;
  timestamp: number;
}

export interface EventFilter {
  minLevel: EventType;
  namespaces?: string[];
  names?: string[];
}

/**
 * Create a filter for events
 */
export function createEventFilter(
  minLevel: EventType = 'info',
  options?: { namespaces?: string[]; names?: string[] }
): EventFilter {
  return {
    minLevel,
    namespaces: options?.namespaces,
    names: options?.names,
  };
}

/**
 * Check if event passes filter
 */
export function eventMatchesFilter(event: LogEvent, filter: EventFilter): boolean {
  // Check level
  const levels: EventType[] = ['debug', 'info', 'warning', 'error', 'progress'];
  const eventLevelIndex = levels.indexOf(event.type);
  const minLevelIndex = levels.indexOf(filter.minLevel);

  if (eventLevelIndex < minLevelIndex) {
    return false;
  }

  // Check namespace
  if (filter.namespaces && filter.namespaces.length > 0) {
    if (!filter.namespaces.includes(event.namespace)) {
      return false;
    }
  }

  // Check event name
  if (filter.names && filter.names.length > 0) {
    if (!event.name || !filter.names.includes(event.name)) {
      return false;
    }
  }

  return true;
}

/**
 * Log level to number mapping
 */
export const LogLevelPriority: Record<EventType, number> = {
  debug: 0,
  info: 1,
  warning: 2,
  error: 3,
  progress: 4,
};

export default LogEvent;