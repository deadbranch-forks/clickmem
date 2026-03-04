/**
 * ClickMem Hook Handler for OpenClaw
 *
 * Injects memory context directly into bootstrapFiles — no disk writes.
 */

import { execFileSync } from "child_process";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CLICKMEM_ROOT = resolve(__dirname, "..");
const MEMORY_BIN = join(CLICKMEM_ROOT, ".venv", "bin", "memory");

const DEBUG = process.env.CLICKMEM_DEBUG === "1" || true;

function log(...args) {
  if (DEBUG) console.log("[clickmem]", ...args);
}

function run(args) {
  log("exec:", MEMORY_BIN, args.join(" "));
  try {
    return execFileSync(MEMORY_BIN, args, {
      encoding: "utf-8",
      timeout: 30000,
    }).trim();
  } catch (err) {
    console.error(`[clickmem] Error: memory ${args.join(" ")}: ${err.message}`);
    return null;
  }
}

/**
 * Handle OpenClaw hook events.
 * Mutates event.context.bootstrapFiles to inject memory content.
 */
export default function handle(event) {
  const eventKey = `${event.type}:${event.action}`;
  log("hook:", eventKey);

  const isBootstrap =
    eventKey === "agent:bootstrap" ||
    eventKey === "command:new" ||
    eventKey === "command:reset";

  if (!isBootstrap) return;

  // Get formatted markdown content from CLI (no file writes)
  const raw = run(["export-context", "--content"]);
  if (!raw) return;

  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    console.error("[clickmem] Failed to parse export-context output");
    return;
  }

  const files = event.context?.bootstrapFiles;
  if (!Array.isArray(files)) {
    log("no bootstrapFiles in context, skipping injection");
    return;
  }

  // Inject MEMORY.md — replace existing or append
  if (data.memory_md) {
    const idx = files.findIndex((f) => f.name === "MEMORY.md");
    const entry = {
      name: "MEMORY.md",
      path: "/clickmem/MEMORY.md",
      content: data.memory_md,
      missing: false,
    };
    if (idx >= 0) {
      files[idx] = entry;
      log("replaced existing MEMORY.md in bootstrapFiles");
    } else {
      files.push(entry);
      log("appended MEMORY.md to bootstrapFiles");
    }
  }

  // Inject daily episodic as memory.md (lowercase — recognized by OpenClaw)
  if (data.daily_md) {
    const idx = files.findIndex((f) => f.name === "memory.md");
    const entry = {
      name: "memory.md",
      path: "/clickmem/memory.md",
      content: data.daily_md,
      missing: false,
    };
    if (idx >= 0) {
      files[idx] = entry;
      log("replaced existing memory.md in bootstrapFiles");
    } else {
      files.push(entry);
      log("appended memory.md to bootstrapFiles");
    }
  }

  event.context.bootstrapFiles = files;
  log("injection done:", files.length, "total bootstrap files");
}
