import { useEffect, useState } from "react";

import { api, setCsrfToken } from "../api/client";

export interface Session {
  csrfToken: string;
}

export function useSession() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  function refresh() {
    setNonce((value) => value + 1);
  }

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .me()
      .then((value) => {
        if (cancelled) {
          return;
        }
        setCsrfToken(value.csrf_token);
        setSession({ csrfToken: value.csrf_token });
      })
      .catch(() => {
        if (!cancelled) {
          setSession(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  return { session, loading, refresh };
}
