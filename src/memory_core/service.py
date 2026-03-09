"""Platform service management — install/uninstall clickmem as a system service.

macOS: ~/Library/LaunchAgents/com.clickmem.server.plist  (launchd user agent)
Linux: ~/.config/systemd/user/clickmem.service           (systemd user unit)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SERVICE_NAME = "clickmem"
_LOG_DIR = Path(os.path.expanduser("~/.openclaw/memory/logs"))

# macOS
_PLIST_NAME = f"com.{_SERVICE_NAME}.server"
_PLIST_DIR = Path(os.path.expanduser("~/Library/LaunchAgents"))
_PLIST_PATH = _PLIST_DIR / f"{_PLIST_NAME}.plist"

# Linux
_SYSTEMD_DIR = Path(os.path.expanduser("~/.config/systemd/user"))
_UNIT_PATH = _SYSTEMD_DIR / f"{_SERVICE_NAME}.service"


def _detect_platform() -> str:
    s = platform.system()
    if s == "Darwin":
        return "macos"
    if s == "Linux":
        return "linux"
    raise RuntimeError(f"Unsupported platform: {s}")


def _find_memory_bin() -> str:
    """Find the absolute path of the ``memory`` CLI binary."""
    # 1) Same prefix as the running Python
    candidate = Path(sys.prefix) / "bin" / "memory"
    if candidate.is_file():
        return str(candidate)
    # 2) shutil.which
    found = shutil.which("memory")
    if found:
        return found
    raise FileNotFoundError(
        "Cannot find 'memory' binary. "
        "Make sure clickmem is installed: pip install clickmem[server]"
    )


# ---------------------------------------------------------------------------
# macOS — launchd
# ---------------------------------------------------------------------------

def _render_plist(memory_bin: str, host: str, port: int) -> str:
    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>{_PLIST_NAME}</string>

        <key>ProgramArguments</key>
        <array>
            <string>{memory_bin}</string>
            <string>serve</string>
            <string>--host</string>
            <string>{host}</string>
            <string>--port</string>
            <string>{port}</string>
        </array>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <dict>
            <key>SuccessfulExit</key>
            <false/>
        </dict>

        <key>ThrottleInterval</key>
        <integer>5</integer>

        <key>StandardOutPath</key>
        <string>{_LOG_DIR / "server.log"}</string>
        <key>StandardErrorPath</key>
        <string>{_LOG_DIR / "server.log"}</string>

        <key>EnvironmentVariables</key>
        <dict>
            <key>PATH</key>
            <string>{os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")}</string>
        </dict>
    </dict>
    </plist>
    """)


def _macos_install(host: str, port: int) -> str:
    memory_bin = _find_memory_bin()
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PLIST_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotent: unload old service if present
    if _PLIST_PATH.exists():
        subprocess.run(
            ["launchctl", "unload", str(_PLIST_PATH)],
            capture_output=True,
        )

    plist = _render_plist(memory_bin, host, port)
    _PLIST_PATH.write_text(plist)

    subprocess.run(
        ["launchctl", "load", "-w", str(_PLIST_PATH)],
        check=True, capture_output=True,
    )
    return str(_PLIST_PATH)


def _macos_uninstall() -> bool:
    if not _PLIST_PATH.exists():
        return False
    subprocess.run(
        ["launchctl", "unload", str(_PLIST_PATH)],
        capture_output=True,
    )
    _PLIST_PATH.unlink(missing_ok=True)
    return True


def _macos_start():
    if not _PLIST_PATH.exists():
        raise RuntimeError("Service not installed. Run: memory service install")
    subprocess.run(
        ["launchctl", "load", "-w", str(_PLIST_PATH)],
        check=True, capture_output=True,
    )


def _macos_stop():
    subprocess.run(
        ["launchctl", "unload", str(_PLIST_PATH)],
        capture_output=True,
    )


def _macos_status() -> dict:
    result = subprocess.run(
        ["launchctl", "list", _PLIST_NAME],
        capture_output=True, text=True,
    )
    installed = _PLIST_PATH.exists()
    running = result.returncode == 0
    pid = None
    if running:
        for line in result.stdout.splitlines():
            if '"PID"' in line or "PID" in line:
                parts = line.strip().rstrip(";").split()
                for p in parts:
                    if p.isdigit():
                        pid = int(p)
                        break
    return {"installed": installed, "running": running, "pid": pid,
            "config": str(_PLIST_PATH) if installed else None}


# ---------------------------------------------------------------------------
# Linux — systemd user unit
# ---------------------------------------------------------------------------

def _render_unit(memory_bin: str, host: str, port: int) -> str:
    return textwrap.dedent(f"""\
    [Unit]
    Description=ClickMem — unified memory server for AI coding agents
    After=network.target

    [Service]
    Type=simple
    ExecStart={memory_bin} serve --host {host} --port {port}
    Restart=on-failure
    RestartSec=5
    Environment=PATH={os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")}

    StandardOutput=append:{_LOG_DIR / "server.log"}
    StandardError=append:{_LOG_DIR / "server.log"}

    [Install]
    WantedBy=default.target
    """)


def _linux_install(host: str, port: int) -> str:
    memory_bin = _find_memory_bin()
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotent: stop old service if present
    if _UNIT_PATH.exists():
        subprocess.run(["systemctl", "--user", "stop", _SERVICE_NAME],
                       capture_output=True)

    unit = _render_unit(memory_bin, host, port)
    _UNIT_PATH.write_text(unit)

    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   check=True, capture_output=True)
    subprocess.run(["systemctl", "--user", "enable", _SERVICE_NAME],
                   check=True, capture_output=True)
    subprocess.run(["systemctl", "--user", "start", _SERVICE_NAME],
                   check=True, capture_output=True)
    return str(_UNIT_PATH)


def _linux_uninstall() -> bool:
    if not _UNIT_PATH.exists():
        return False
    subprocess.run(["systemctl", "--user", "stop", _SERVICE_NAME],
                   capture_output=True)
    subprocess.run(["systemctl", "--user", "disable", _SERVICE_NAME],
                   capture_output=True)
    _UNIT_PATH.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   capture_output=True)
    return True


def _linux_start():
    if not _UNIT_PATH.exists():
        raise RuntimeError("Service not installed. Run: memory service install")
    subprocess.run(["systemctl", "--user", "start", _SERVICE_NAME],
                   check=True, capture_output=True)


def _linux_stop():
    subprocess.run(["systemctl", "--user", "stop", _SERVICE_NAME],
                   capture_output=True)


def _linux_status() -> dict:
    installed = _UNIT_PATH.exists()
    result = subprocess.run(
        ["systemctl", "--user", "is-active", _SERVICE_NAME],
        capture_output=True, text=True,
    )
    running = result.stdout.strip() == "active"
    pid = None
    if running:
        pid_result = subprocess.run(
            ["systemctl", "--user", "show", _SERVICE_NAME, "--property=MainPID"],
            capture_output=True, text=True,
        )
        for line in pid_result.stdout.splitlines():
            if line.startswith("MainPID="):
                v = line.split("=", 1)[1].strip()
                if v.isdigit() and v != "0":
                    pid = int(v)
    return {"installed": installed, "running": running, "pid": pid,
            "config": str(_UNIT_PATH) if installed else None}


# ---------------------------------------------------------------------------
# Public API — platform-dispatching
# ---------------------------------------------------------------------------

def install(host: str = "0.0.0.0", port: int = 9527) -> str:
    """Install and start the service. Returns the config file path."""
    plat = _detect_platform()
    if plat == "macos":
        return _macos_install(host, port)
    return _linux_install(host, port)


def uninstall() -> bool:
    """Stop and remove the service. Returns True if it was installed."""
    plat = _detect_platform()
    if plat == "macos":
        return _macos_uninstall()
    return _linux_uninstall()


def start():
    """Start the service (must be installed first)."""
    plat = _detect_platform()
    if plat == "macos":
        _macos_start()
    else:
        _linux_start()


def stop():
    """Stop the service."""
    plat = _detect_platform()
    if plat == "macos":
        _macos_stop()
    else:
        _linux_stop()


def status() -> dict:
    """Return service status: installed, running, pid, config path."""
    plat = _detect_platform()
    if plat == "macos":
        return _macos_status()
    return _linux_status()


def log_path() -> Path:
    """Return the log file path."""
    return _LOG_DIR / "server.log"
