/**
 * ClickMem Hook Handler for OpenClaw
 *
 * Bridges OpenClaw events to the clickmem CLI:
 * - On bootstrap/new/reset: exports memory context to the workspace
 */

import { execFileSync } from "child_process";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CLICKMEM_ROOT = resolve(__dirname, "..");
const MEMORY_BIN = join(CLICKMEM_ROOT, ".venv", "bin", "memory");

const DEBUG = process.env.CLICKMEM_DEBUG === "1" || true; // always on for now

function log(...args) {
  if (DEBUG) console.log("[clickmem]", ...args);
}

function run(args, options = {}) {
  log("exec:", MEMORY_BIN, args.join(" "));
  try {
    const result = execFileSync(MEMORY_BIN, args, {
      encoding: "utf-8",
      timeout: 30000,
      ...options,
    });
    log("exec result:", result.trim().slice(0, 200));
    return result.trim();
  } catch (err) {
    console.error(`[clickmem] Error running: memory ${args.join(" ")}`);
    console.error(`[clickmem] ${err.message}`);
    return null;
  }
}

/**
 * Handle OpenClaw hook events.
 * @param {object} event - Hook event: {type, action, sessionKey, context, timestamp, messages}
 */
export default function handle(event) {
  const eventKey = `${event.type}:${event.action}`;
  log("hook called:", eventKey, "sessionKey:", event.sessionKey);
  log("context:", JSON.stringify(event.context || {}, null, 2).slice(0, 500));

  const workspacePath =
    event.context?.workspaceDir || event.context?.workspacePath || "";

  log("workspacePath:", workspacePath || "(empty)");

  switch (eventKey) {
    case "agent:bootstrap":
    case "command:new":
    case "command:reset":
      if (workspacePath) {
        log("exporting context to", workspacePath);
        const result = run(["export-context", workspacePath, "--json"]);
        log("export done:", result);
      } else {
        log("skipping export: no workspacePath in context");
      }
      break;

    default:
      log("unhandled event:", eventKey);
      break;
  }
}
