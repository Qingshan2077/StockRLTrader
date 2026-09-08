import { useRef, useState } from "react";
import { ApiError, request } from "./client";
import type { SubmissionResult } from "./contracts";

export interface PendingSubmission {
  key: string;
  path: string;
  payload: string;
  createdAt: string;
}
export function createPending(path: string, body: unknown): PendingSubmission {
  return {
    key: crypto.randomUUID(),
    path,
    payload: JSON.stringify(body),
    createdAt: new Date().toISOString(),
  };
}
export function readStored<T>(key: string): T | null {
  try {
    const value = sessionStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : null;
  } catch {
    return null;
  }
}
export function saveStored(key: string, value: unknown): void {
  sessionStorage.setItem(key, JSON.stringify(value));
}
export function removeStored(key: string): void {
  sessionStorage.removeItem(key);
}
export function useSubmission(
  storageKey: string,
  onSuccess: (result: SubmissionResult) => void,
) {
  const [pending, setPending] = useState<PendingSubmission | null>(() => {
    const value = readStored<PendingSubmission>(storageKey);
    if (
      !value ||
      typeof value.key !== "string" ||
      typeof value.path !== "string" ||
      typeof value.payload !== "string"
    )
      return null;
    try {
      JSON.parse(value.payload);
      return value;
    } catch {
      return null;
    }
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const inFlight = useRef(false);
  const pendingRef = useRef(pending);
  async function send(attempt: PendingSubmission) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);
    try {
      // Persist before the network call: an uncertain request always retains the exact body and key.
      saveStored(storageKey, attempt);
      pendingRef.current = attempt;
      setPending(attempt);
      const result = await request<SubmissionResult>(attempt.path, {
        method: "POST",
        body: attempt.payload,
        headers: { "Idempotency-Key": attempt.key },
      });
      removeStored(storageKey);
      pendingRef.current = null;
      setPending(null);
      onSuccess(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error("提交失败。"));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }
  return {
    pending,
    busy,
    error,
    start: (path: string, body: unknown) => {
      if (!pendingRef.current && !inFlight.current)
        void send(createPending(path, body));
    },
    retry: () => {
      if (pendingRef.current) void send(pendingRef.current);
    },
    abandon: () => {
      if (inFlight.current) return;
      removeStored(storageKey);
      pendingRef.current = null;
      setPending(null);
      setError(null);
    },
    uncertain:
      !error ||
      !(error instanceof ApiError) ||
      error.status === 0 ||
      error.status >= 500,
  };
}
