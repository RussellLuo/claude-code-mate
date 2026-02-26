#!/usr/bin/env python3
"""
Claude Code Mate (CCM) - A companion tool for Claude Code, enabling flexible LLM integration through LiteLLM proxy.
"""

import argparse
import os
from pathlib import Path
import platform
import signal
import time
import shlex
import subprocess
import sys
from typing import Optional, Dict, Any

from jinja2 import Template
import psutil
import webbrowser
import yaml

from .constants import WORK_DIR, CONFIG_PATH, ensure_work_dir
from .postgres import Postgres


class Platform:
    SYSTEM = platform.system()
    IS_WINDOWS = SYSTEM == "Windows"
    IS_LINUX = SYSTEM == "Linux"
    IS_MAC = SYSTEM == "Darwin"


class ProcessManager:
    """Manages background processes with PID file tracking"""

    def __init__(self, name: str):
        self.name = name
        self.work_dir = WORK_DIR
        self.pid_file = self.work_dir / f"{name}.pid"
        self.log_file = self.work_dir / f"{name}.log"

    def is_running(self) -> bool:
        """Check if the service is currently running"""
        return self.get_pid() is not None

    def _read_pid_from_file(self) -> Optional[int]:
        """Read PID from file, return None if file doesn't exist or invalid"""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, "r") as f:
                content = f.read().strip()
                return int(content) if content else None
        except (OSError, ValueError):
            return None

    def _is_valid_litellm_process(self, pid: int) -> bool:
        """Check if PID is a valid running litellm process"""
        try:
            if not psutil.pid_exists(pid):
                return False
            process = psutil.Process(pid)
            cmdline = process.cmdline()
            return any("litellm" in cmd for cmd in cmdline)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            return False

    def _find_litellm_process(self, update_pid_file: bool = False) -> Optional[int]:
        """Find running litellm process and return its PID"""
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.info["cmdline"]
                    if cmdline and any("litellm" in cmd for cmd in cmdline):
                        if update_pid_file:
                            current_pid = self._read_pid_from_file()
                            if current_pid != proc.info["pid"]:
                                print(
                                    f"Found orphaned litellm process (PID: {proc.info['pid']}), updating PID file..."
                                )
                                self.save_pid(proc.info["pid"])
                        return proc.info["pid"]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except psutil.Error:
            pass
        return None

    def save_pid(self, pid: int) -> None:
        """Save process ID to file"""
        with open(self.pid_file, "w") as f:
            f.write(str(pid))

    def get_pid(self) -> Optional[int]:
        """Get the current service PID"""
        pid = self._read_pid_from_file()

        if pid and self._is_valid_litellm_process(pid):
            return pid

        if pid:
            self._cleanup_pid_file()

        return self._find_litellm_process(update_pid_file=True)

    def _cleanup_pid_file(self) -> None:
        """Remove PID file"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except OSError:
            pass

    def stop_service(self) -> None:
        """Stop the running service in a cross-platform way"""
        pid = self.get_pid()
        try:
            # Try to terminate gracefully first
            process = psutil.Process(pid)
            process.terminate()

            # Wait for process to terminate
            self._force_kill_if_needed(process, pid)
        except psutil.NoSuchProcess:
            pass
        finally:
            self._cleanup_pid_file()

    def _force_kill_if_needed(self, process: psutil.Process, pid: int) -> None:
        """Helper method to handle process termination with timeout and force kill"""

        def force_kill():
            if Platform.IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except psutil.TimeoutExpired:
                    pass

        try:
            gone, alive = psutil.wait_procs([process], timeout=5)
            if alive:
                force_kill()
        except psutil.TimeoutExpired:
            force_kill()

    def _read_file_lines(
        self, file_path: Path, num_lines: Optional[int] = None
    ) -> list[str]:
        """Read lines from file, optionally limiting to last N lines"""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                return lines[-num_lines:] if num_lines else lines
        except Exception:
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get service status information"""
        return {
            "name": self.name,
            "running": self.is_running(),
            "pid": self.get_pid(),
            "pid_file": str(self.pid_file),
            "log_file": str(self.log_file),
            "work_dir": str(self.work_dir),
        }


class BackgroundService:
    """Main service class for running commands in background"""

    def __init__(self, command: list):
        if not command:
            raise ValueError("Command cannot be empty")

        self.command = command
        # Extract name from first command, remove path if present
        self.name = Path(command[0]).name
        self.process_manager = ProcessManager(self.name)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown in a cross-platform way"""

        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up...")
            self.process_manager._cleanup_pid_file()
            sys.exit(0)

        # Register signals with cross-platform compatibility
        # SIGINT (Ctrl+C) works on all platforms
        signal.signal(signal.SIGINT, signal_handler)

        # SIGTERM is available on all platforms, but handled differently
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows specific signals
        if Platform.IS_WINDOWS and hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal_handler)

    def start(self) -> bool:
        """Start the configured command as a background service"""
        if self.process_manager.is_running():
            print("✅ Service is already running")
            print("Use 'ccm status' to view details")
            return True

        # Handle command parsing - if it's a single string with spaces, split it
        if len(self.command) == 1 and " " in self.command[0]:
            parsed_command = shlex.split(self.command[0])
        else:
            parsed_command = self.command

        try:
            # Cross-platform background process creation
            with open(self.process_manager.log_file, "a") as log_file:
                # Platform-specific process creation
                if Platform.IS_WINDOWS:
                    # Windows: use CREATE_NEW_PROCESS_GROUP flag
                    # and DETACHED_PROCESS to detach from console
                    DETACHED_PROCESS = 0x00000008
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

                    process = subprocess.Popen(
                        parsed_command,
                        stdout=log_file,
                        stderr=log_file,
                        stdin=subprocess.DEVNULL,
                        creationflags=flags,
                        cwd=os.getcwd(),
                    )
                else:
                    # Unix: use start_new_session=True to create new process group
                    process = subprocess.Popen(
                        parsed_command,
                        stdout=log_file,
                        stderr=log_file,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,  # Detach from terminal session
                        cwd=os.getcwd(),
                    )

                # Save PID and exit parent process
                self.process_manager.save_pid(process.pid)
                print("✅ Service started successfully")
                print("Use 'ccm status' to view details")
                print("\n💡 To use with Claude Code, set these environment variables:")
                print("export ANTHROPIC_BASE_URL=http://0.0.0.0:4000")
                print(
                    f"export ANTHROPIC_AUTH_TOKEN={ConfigManager().litellm_master_key}"
                )
                return True

        except Exception as e:
            print("❌ Service failed to start")
            return False

    def stop(self, cleanup_db: bool = True) -> bool:
        """Stop the background service"""
        if not self.process_manager.is_running():
            print("Service is not running")
            return False
        try:
            self.process_manager.stop_service()
            if cleanup_db:
                Postgres.cleanup()
            print("✅ Service stopped successfully")
            return True
        except Exception as e:
            print(f"❌ Service failed to stop: {e}")
            return False

    def restart(self) -> bool:
        """Restart the service with the configured command"""
        print("Stopping service...")
        self.stop(cleanup_db=False)

        # Wait a moment for cleanup
        time.sleep(1)

        print("Starting service...")
        return self.start()

    def status(self) -> None:
        """Show service status"""
        status = self.process_manager.get_status()

        print(f"📊 {status['name'].upper()} Service Status:")
        print(f"  Status: {'🟢 Running' if status['running'] else '🔴 Stopped'}")
        if status["pid"]:
            print(f"  PID: {status['pid']}")
        print(f"  Work Dir: {status['work_dir']}")
        print(f"  PID File: {status['pid_file']}")
        print(f"  Log File: {status['log_file']}")

        # Show recent logs if available
        if Path(status["log_file"]).exists():
            print(f"\n📝 Recent logs (last 10 lines):")
            lines = self.process_manager._read_file_lines(Path(status["log_file"]), 10)
            if lines:
                for line in lines:
                    print(f"    {line.rstrip()}")
            else:
                print("    Error reading logs")

    def _read_file_lines(
        self, file_path: Path, num_lines: Optional[int] = None
    ) -> list[str]:
        """Read lines from file, optionally limiting to last N lines"""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                return lines[-num_lines:] if num_lines else lines
        except Exception:
            return []

    def logs(self, follow: bool = False, lines: int = 50) -> None:
        """Show service logs in a cross-platform way"""
        log_file = self.process_manager.log_file

        if not log_file.exists():
            print("No log file found")
            return

        if follow:
            # Cross-platform log following implementation
            print(f"Following logs from {log_file} (Ctrl+C to stop)...\n")
            try:
                # Get initial file size
                file_size = log_file.stat().st_size

                # First show the last N lines
                lines_buffer = self._read_file_lines(log_file, lines)
                for line in lines_buffer:
                    print(line.rstrip())

                # Then follow the file for new content
                while True:
                    # Small sleep to reduce CPU usage
                    time.sleep(0.1)

                    # Check new file size
                    current_size = log_file.stat().st_size

                    if current_size > file_size:
                        # File grew, read new content
                        with open(log_file, "r") as f:
                            f.seek(file_size)
                            new_content = f.read()
                            print(new_content, end="")

                        file_size = current_size
            except KeyboardInterrupt:
                print("\nStopped following logs")
        else:
            # Show last N lines
            lines = self._read_file_lines(log_file, lines)
            for line in lines:
                print(line.rstrip())


class ConfigManager:
    """Manages configuration file creation, editing, and initialization"""

    def __init__(self, path: str = CONFIG_PATH, database_url: str = ""):
        self.config_path = path
        self.default_config = Template("""\
# LiteLLM Proxy Configuration

general_settings:
  master_key: sk-1234567890 # Generate a secure key
  {%- if database_url %}
  database_url: {{ database_url }}
  store_model_in_db: true
  {%- endif %}

{% if database_url -%}
# Uncomment the following section to configure models if you like YAML-based
# model management, or go to the Admin UI for UI-based model management.
#model_list:
#  # OpenRouter Example
#  #
#  # For details, see:
#  # - https://docs.litellm.ai/docs/tutorials/claude_responses_api
#  # - https://docs.litellm.ai/docs/providers
#  - model_name: claude-3.5-haiku
#    litellm_params:
#      model: openrouter/anthropic/claude-3.5-haiku
#      api_key: os.environ/OPENROUTER_API_KEY
#      api_base: https://openrouter.ai/api/v1
{% else -%}
model_list:
  # OpenRouter Example
  #
  # For details, see:
  # - https://docs.litellm.ai/docs/tutorials/claude_responses_api
  # - https://docs.litellm.ai/docs/providers
  - model_name: claude-3.5-haiku
    litellm_params:
      model: openrouter/anthropic/claude-3.5-haiku
      api_key: os.environ/OPENROUTER_API_KEY
      api_base: https://openrouter.ai/api/v1
{%- endif %}
""").render(database_url=database_url)

    @property
    def litellm_master_key(self) -> str:
        """Read litellm_settings.master_key from config.yaml file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("litellm_settings", {})["master_key"]
        except Exception:
            return "sk-1234567890"

    def open_in_editor(self) -> bool:
        """Open the config file in an editor, similar to git config -e"""
        try:
            # Get editor from environment variables, fallback to common editors
            editor = os.getenv("EDITOR")
            if not editor:
                # Platform-specific default editors
                if Platform.IS_WINDOWS:
                    editor = "notepad"
                else:  # Mac, Linux and others
                    # Try vim and nano for Unix-like systems
                    for candidate in ["vim", "nano"]:
                        try:
                            subprocess.run(
                                ["which", candidate],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            editor = candidate
                            break
                        except subprocess.CalledProcessError:
                            continue
                    else:
                        editor = "nano"  # fallback

            # Open the config file with the chosen editor
            print(f"🔧 Opening config file with {editor}...")
            result = subprocess.run([editor, str(self.config_path)], check=True)
            return result.returncode == 0

        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def initialize(self) -> bool:
        """Initialize configuration file if it doesn't exist"""
        if self.config_path.exists():
            return True

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(self.default_config)

            # Auto-open the config file in editor for first-time setup
            if self.open_in_editor():
                print(f"📝 Configuration file saved: {self.config_path}")
            else:
                print(f"💡 Please manually edit the config file: {self.config_path}")

            return True

        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
            return False


def main():
    """Main CLI interface for LiteLLM proxy management"""
    parser = argparse.ArgumentParser(
        prog="ccm",
        description="A companion tool for Claude Code, enabling flexible LLM integration through LiteLLM proxy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=Template("""
Examples:
  ccm start
  ccm stop
  ccm restart
  ccm status
  ccm logs
  ccm logs -f -n 100
  {%- if has_ui %}
  ccm ui
  {%- endif %}

This tool manages a LiteLLM proxy running with: litellm --config ~/.claude-code-mate/config.yaml
""").render(has_ui=Postgres.has_ui()),
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the LiteLLM proxy in background",
        description="Start the LiteLLM proxy as a background process. The proxy will run with the configuration from ~/.claude-code-mate/config.yaml",
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the running LiteLLM proxy",
        description="Stop the currently running LiteLLM proxy. This will terminate the background process gracefully.",
    )

    # Restart command
    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart the LiteLLM proxy",
        description="Stop the current proxy if running, then start it again with the latest configuration.",
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show current proxy status",
        description="Display detailed information about the LiteLLM proxy including running status, PID, log file location, and recent log entries.",
    )

    # Logs command
    logs_parser = subparsers.add_parser(
        "logs",
        help="Show proxy logs",
        description="Display log output from the LiteLLM proxy. Use --follow to watch logs in real-time, or --lines to control how many lines to show.",
    )
    logs_parser.add_argument(
        "--follow", "-f", action="store_true", help="Follow logs in real-time"
    )
    logs_parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=50,
        help="Number of log lines to show (default: 50)",
    )

    if Postgres.has_ui():
        # UI command
        ui_parser = subparsers.add_parser(
            "ui",
            help="Open LiteLLM UI in browser",
            description="Open the LiteLLM web UI at http://0.0.0.0:4000/ui in your default browser.",
        )
        ui_parser.add_argument(
            "--regenerate",
            action="store_true",
            default=False,
            help="Force regenerate Prisma client from LiteLLM schema (default: False)",
        )

    args = parser.parse_args()

    # Ensure work directory exists
    ensure_work_dir()

    # Initialize database and get the connection URL
    regenerate = getattr(args, "regenerate", False)
    database_url = Postgres.initialize(regenerate=regenerate)

    # Initialize config file if it doesn't exist
    config_manager = ConfigManager(database_url=database_url)
    if not config_manager.initialize():
        sys.exit(1)

    # litellm --config ~/.claude-code-mate/config.yaml
    litellm_command = ["litellm", "--config", str(CONFIG_PATH)]
    service = BackgroundService(command=litellm_command)

    if args.command == "start":
        success = service.start()
        sys.exit(0 if success else 1)

    elif args.command == "stop":
        success = service.stop()
        sys.exit(0 if success else 1)

    elif args.command == "restart":
        success = service.restart()
        sys.exit(0 if success else 1)

    elif args.command == "status":
        service.status()

    elif args.command == "logs":
        service.logs(follow=args.follow, lines=args.lines)

    elif args.command == "ui":
        ui_url = "http://0.0.0.0:4000/ui"
        print(f"🌐 Opening LiteLLM UI: {ui_url}")
        try:
            webbrowser.open(ui_url)
        except Exception as e:
            print(f"❌ Failed to open browser: {e}")
            print(f"💡 Please manually open: {ui_url}")


if __name__ == "__main__":
    main()
