import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export function resolveApiTarget(value: string | undefined): string {
  const target = new URL(value?.trim() || "http://127.0.0.1:8000");
  if (
    target.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "[::1]"].includes(target.hostname) ||
    target.username || target.password ||
    target.pathname !== "/" || target.search || target.hash
  ) {
    throw new Error("STOCKRL_API_TARGET 必须是本机 HTTP API 地址，例如 http://127.0.0.1:8081");
  }
  return target.origin;
}

const envDir = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig(({ mode }) => ({
  envDir,
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": resolveApiTarget(loadEnv(mode, envDir, "STOCKRL_").STOCKRL_API_TARGET) },
  },
  build: { outDir: "../api/static", emptyOutDir: true, sourcemap: false },
}));
