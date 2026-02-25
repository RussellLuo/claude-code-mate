"""
Database management for Claude Code Mate.

Provides PostgreSQL server management and Prisma client generation
for LiteLLM proxy's Admin UI features.

Requirements:
    Optional 'ui' dependencies: pgserver, prisma
"""

import importlib.util
from pathlib import Path
import platform
import subprocess

try:
    import pgserver
except ImportError:
    pgserver = None

from constants import WORK_DIR


class Postgres:
    """Database operations for PostgreSQL server and Prisma client management."""

    data_dir = WORK_DIR / "postgres_data"

    @classmethod
    def _get_litellm_schema_path(cls) -> Path:
        """Locate LiteLLM's schema.prisma file."""
        # Try to find the LiteLLM package location
        spec = importlib.util.find_spec("litellm")
        if not (spec and spec.origin):
            raise ImportError("LiteLLM package not found.")

        litellm_path = Path(spec.origin).parent
        schema_path = litellm_path / "proxy" / "schema.prisma"
        if not schema_path.exists():
            raise FileNotFoundError("LiteLLM schema.prisma file not found.")

        return schema_path

    @classmethod
    def _get_prisma_command(cls) -> str:
        """Find the appropriate Prisma command for current platform."""
        if platform.system() == "Windows":
            # Try Windows-specific Prisma commands
            for cmd in ["prisma.cmd", "prisma.exe"]:
                try:
                    subprocess.run([cmd, "--version"], capture_output=True, check=True)
                    return cmd
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue

        # Default for Unix-like systems
        return "prisma"

    @classmethod
    def _generate_prisma_client(cls, verbose: bool = False) -> bool:
        """Generate Prisma client from LiteLLM schema."""
        try:
            schema_path = cls._get_litellm_schema_path()
            prisma_cmd = cls._get_prisma_command()
            result = subprocess.run(
                [prisma_cmd, "generate"],
                capture_output=True,
                text=True,
                cwd=schema_path.parent,
            )

            if result.returncode == 0:
                if verbose and result.stdout:
                    print(f"Output: {result.stdout}")
                return True
            else:
                print(f"❌ Prisma generate failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False

        except FileNotFoundError:
            print(
                '❌ Prisma CLI not found. Please install with: pip install "claude-code-mate[ui]"'
            )
            return False
        except Exception as e:
            print(f"❌ Unexpected error during Prisma generation: {e}")
            return False

    @classmethod
    def has_ui(cls) -> bool:
        """Check if Admin UI is available."""
        return pgserver is not None

    @classmethod
    def initialize(cls) -> str:
        """Initialize database with Prisma client generation and PostgreSQL setup."""
        if not cls.has_ui():
            return ""

        # Check if Prisma client is already available
        try:
            from prisma import Prisma
        except RuntimeError as e:
            # Example error message:
            #   The Client hasn't been generated yet, you must run `prisma generate` before you can use the client.
            if "prisma generate" in str(e):
                cls._generate_prisma_client()

        pg = pgserver.get_server(cls.data_dir, cleanup_mode=None)
        # Manually construct URI with proper host:port for Prisma compatibility.
        #
        # On Linux/Unix, `pg.get_uri()` respects the unix socket directory and returns
        # "postgresql://postgres:@/postgres?host={cls.data_dir}", which will cause an
        # error when used by Prisma:
        #   Error: P1013: The provided database string is invalid. empty host in database URL.
        socket_dir = pg.get_postmaster_info().socket_dir
        if socket_dir is not None:
            return f"postgresql://postgres:@localhost:5432/postgres?host={socket_dir}"
        else:
            return pg.get_uri()

    @classmethod
    def cleanup(cls, mode: str = "stop") -> None:
        """Clean up database resources."""
        if not cls.has_ui():
            return

        pg = pgserver.PostgresServer(cls.data_dir, cleanup_mode=mode)
        try:
            pg.cleanup()
        except Exception as e:
            print(f"⚠️  Warning during database cleanup: {e}")
