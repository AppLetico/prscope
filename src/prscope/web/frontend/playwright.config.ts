import { defineConfig, devices } from "@playwright/test";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..", "..", "..");
const fixtureConfig = path.resolve(__dirname, "e2e", "fixtures");
const venvPython = path.join(repoRoot, ".venv", "bin", "python3");
const pythonExe = fs.existsSync(venvPython) ? venvPython : "python3";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8420",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run build && cd "${repoRoot}" && PYTHONPATH="${repoRoot}/src" PRSCOPE_CONFIG_ROOT="${fixtureConfig}" "${pythonExe}" -m uvicorn prscope.web.server:create_server_app --factory --host 127.0.0.1 --port 8420`,
    cwd: __dirname,
    url: "http://127.0.0.1:8420/health",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
