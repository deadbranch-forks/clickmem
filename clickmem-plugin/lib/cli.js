import { execFile, execFileSync } from "child_process";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const LIB_DIR = dirname(__filename);
const PLUGIN_DIR = dirname(LIB_DIR);
const CLICKMEM_ROOT = resolve(PLUGIN_DIR, "..");
const MEMORY_BIN = join(CLICKMEM_ROOT, ".venv", "bin", "memory");

export function runSync(args) {
  return execFileSync(MEMORY_BIN, args, { encoding: "utf-8", timeout: 30000 }).trim();
}

// Mutex to serialize CLI calls (chDB only allows one process per db path)
let _queue = Promise.resolve();

export function runAsync(args) {
  const job = _queue.then(() => new Promise((resolve, reject) => {
    execFile(MEMORY_BIN, args, { timeout: 30000 }, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout.trim());
    });
  }));
  // Chain next job regardless of success/failure
  _queue = job.catch(() => {});
  return job;
}
