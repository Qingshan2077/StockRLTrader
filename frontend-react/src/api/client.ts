import { useQuery } from "@tanstack/react-query";
import { useRef } from "react";
import type { Capabilities, JobDetail } from "./contracts";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public requestId: string | null,
    public details: unknown = null,
  ) {
    super(message);
  }
}
export async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...init,
      headers,
      credentials: "same-origin",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new ApiError(
      "无法确认服务器是否收到请求。请检查连接后重试。",
      0,
      "NETWORK_UNKNOWN",
      null,
    );
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      "服务器返回了无法识别的响应。",
      response.status,
      "INVALID_RESPONSE",
      response.headers.get("X-Request-ID"),
    );
  }
  if (!response.ok) {
    const envelope = object(body);
    const error = object(envelope.error);
    throw new ApiError(
      typeof error.message === "string" ? error.message : "请求失败，请重试。",
      response.status,
      typeof error.code === "string" ? error.code : "HTTP_ERROR",
      typeof envelope.request_id === "string"
        ? envelope.request_id
        : response.headers.get("X-Request-ID"),
      error.details,
    );
  }
  return body as T;
}
export function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
export function search(
  values: Record<string, string | number | null | undefined>,
): string {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "")
      params.set(key, String(value));
  });
  return params.size ? `?${params}` : "";
}
export const terminal = (status?: string) =>
  ["succeeded", "failed", "cancelled", "interrupted"].includes(status ?? "");
export function useCapabilities() {
  return useQuery({
    queryKey: ["capabilities"],
    queryFn: ({ signal }) => request<Capabilities>("/capabilities", { signal }),
    staleTime: 30000,
    refetchInterval: 30000,
  });
}
export function useJob(id: string | null) {
  const interval = usePollInterval();
  return useQuery({
    queryKey: ["job", id],
    queryFn: ({ signal }) => request<JobDetail>(`/jobs/${id}`, { signal }),
    enabled: !!id,
    retry: false,
    refetchInterval: (query) =>
      terminal(query.state.data?.status) ? false : interval(query.state),
    refetchIntervalInBackground: false,
  });
}
export function usePollInterval() {
  const tracker = useRef({ lastError: 0, errors: 0 });
  return (state: {
    errorUpdatedAt: number;
    dataUpdatedAt: number;
    error: unknown;
  }) => {
    if (!state.error || state.dataUpdatedAt >= state.errorUpdatedAt)
      tracker.current.errors = 0;
    else if (state.errorUpdatedAt !== tracker.current.lastError) {
      tracker.current.lastError = state.errorUpdatedAt;
      tracker.current.errors++;
    }
    return Math.min(10000, 2000 * 2 ** Math.min(3, tracker.current.errors));
  };
}
export function safeDownload(value: string): string | null {
  return /^\/api\/v1\/artifacts\/[0-9a-f-]{36}\/download$/i.test(value)
    ? value
    : null;
}
export async function downloadArtifact(
  url: string,
  filename: string,
): Promise<void> {
  if (!safeDownload(url)) throw new Error("下载地址无效。");
  let response: Response;
  try {
    response = await fetch(url, { credentials: "same-origin" });
  } catch {
    throw new ApiError(
      "下载连接中断，请重试。",
      0,
      "DOWNLOAD_CONNECTION",
      null,
    );
  }
  if (!response.ok) {
    const body = object(await response.json().catch(() => null));
    const error = object(body.error);
    throw new ApiError(
      typeof error.message === "string" ? error.message : "产物无法下载。",
      response.status,
      typeof error.code === "string" ? error.code : "DOWNLOAD_FAILED",
      typeof body.request_id === "string"
        ? body.request_id
        : response.headers.get("X-Request-ID"),
      error.details,
    );
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename.replaceAll("/", "__");
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}
