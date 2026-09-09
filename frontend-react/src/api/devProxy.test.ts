import { describe, expect, it } from "vitest";
import { resolveApiTarget } from "../../vite.config";

describe("development API target", () => {
  it("keeps the default backend and accepts a separate worktree port", () => {
    expect(resolveApiTarget(undefined)).toBe("http://127.0.0.1:8000");
    expect(resolveApiTarget("http://127.0.0.1:8081")).toBe("http://127.0.0.1:8081");
  });

  it.each(["https://example.org", "http://user:pass@127.0.0.1:8081", "http://127.0.0.1:8081/api"])(
    "rejects a target outside the local API origin: %s", (value) => {
      expect(() => resolveApiTarget(value)).toThrow();
    },
  );
});
