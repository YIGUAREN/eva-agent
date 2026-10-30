import { useEffect, useState } from "react";

import { mergeEvent, parseEventData } from "../api/sse";
import type { RunEvent } from "../api/types";

export function useRunEvents(runId: string, publicView = false) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    setEvents([]);
    setConnected(false);
    if (!runId) {
      return;
    }
    const prefix = publicView ? "/api/public/runs" : "/api/runs";
    const source = new EventSource(`${prefix}/${runId}/events`);
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (message) => {
      const event = parseEventData(message.data);
      if (event) {
        setEvents((current) => mergeEvent(current, event));
      }
    };
    return () => source.close();
  }, [runId, publicView]);

  return { events, connected };
}
