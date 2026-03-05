import { execFile, execFileSync } from "child_process";
import { readFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const LIB_DIR = dirname(__filename);
const PLUGIN_DIR = dirname(LIB_DIR);
const CLICKMEM_ROOT = resolve(PLUGIN_DIR, "..");
const MEMORY_BIN = join(CLICKMEM_ROOT, ".venv", "bin", "memory");

let _llmEnv = {};

// Auto-load LLM config from clickmem-root/llm.json on first import
try {
  const raw = readFileSync(join(CLICKMEM_ROOT, "llm.json"), "utf-8");
  const cfg = JSON.parse(raw);
  if (cfg.baseUrl)  _llmEnv.OPENAI_API_BASE = cfg.baseUrl;
  if (cfg.apiKey)   _llmEnv.OPENAI_API_KEY = cfg.apiKey;
  if (cfg.model)    _llmEnv.CLICKMEM_LLM_MODEL = cfg.model;
} catch { /* no llm.json or parse error — rely on process.env */ }

function execEnv() {
  return { ...process.env, ..._llmEnv };
}

export function runSync(args) {
  return execFileSync(MEMORY_BIN, args, { encoding: "utf-8", timeout: 30000, env: execEnv() }).trim();
}

// Mutex to serialize CLI calls (chDB only allows one process per db path)
let _queue = Promise.resolve();

export function runAsync(args) {
  const job = _queue.then(() => new Promise((resolve, reject) => {
    execFile(MEMORY_BIN, args, { timeout: 30000, env: execEnv() }, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout.trim());
    });
  }));
  // Chain next job regardless of success/failure
  _queue = job.catch(() => {});
  return job;
}
