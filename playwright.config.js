const { defineConfig } = require("@playwright/test");

const PORT = 8765;
const baseURL = `http://127.0.0.1:${PORT}`;

module.exports = defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL,
    trace: "on-first-retry",
  },
  webServer: {
    command: `uv run uvicorn formulatech.api:app --host 127.0.0.1 --port ${PORT}`,
    url: `${baseURL}/api/config`,
    reuseExistingServer: true,
    timeout: 120_000,
    // Point the LLM at a closed port so a missing Ollama falls back immediately.
    // The page renders the deterministic verdict, not the explanation text.
    env: {
      OLLAMA_URL: "http://127.0.0.1:9",
      // Keep e2e audit records out of logs/audit.jsonl.
      FORMULATECH_AUDIT_LOG: "test-results/audit.jsonl",
    },
  },
});
