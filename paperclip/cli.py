"""
Paperclip daemon CLI.

Usage:
  paperclip --authenticate <provider_token>
  paperclip --allocate <0-100>
  paperclip --status
  paperclip --daemon
  paperclip --version
"""
import sys
import time
import logging
import platform
import click
from rich.console import Console
from rich.table import Table
from rich import box

from paperclip import __version__
from paperclip import config, api_client
from paperclip.worker import run_job

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _require_device_token() -> str:
    token = config.device_token()
    if not token:
        console.print("[red]✗[/red] Not authenticated. Run [bold]paperclip --authenticate <token>[/bold] first.")
        sys.exit(1)
    return token


# ─────────────────────────────────────────────────────────────────────────────

@click.command(name="paperclip")
@click.option("--authenticate", metavar="TOKEN", default=None,
              help="Authenticate this device with your provider account token.")
@click.option("--allocate", type=click.IntRange(0, 100), metavar="PCT", default=None,
              help="Set GPU allocation percentage (0–100).")
@click.option("--status", "show_status", is_flag=True, default=False,
              help="Show current device status and config.")
@click.option("--daemon", "run_daemon", is_flag=True, default=False,
              help="Start the daemon (polls for jobs continuously).")
@click.option("--version", "show_version", is_flag=True, default=False,
              help="Print the daemon version and exit.")
def cli(authenticate, allocate, show_status, run_daemon, show_version):
    """Paperclip — GPU provider daemon."""

    if show_version:
        console.print(f"paperclip v{__version__}")
        return

    if authenticate:
        _cmd_authenticate(authenticate)
        return

    if allocate is not None:
        _cmd_allocate(allocate)
        return

    if show_status:
        _cmd_status()
        return

    if run_daemon:
        _cmd_daemon()
        return

    # No flag given → show help
    ctx = click.get_current_context()
    click.echo(ctx.get_help())


# ── Commands ──────────────────────────────────────────────────────────────────

def _cmd_authenticate(provider_token: str) -> None:
    console.print(f"[dim]Connecting to {config.api_url()}…[/dim]")
    try:
        # Try to detect GPU model
        gpu_model = _detect_gpu()
        device_name = platform.node()

        result = api_client.authenticate_device(
            provider_token=provider_token,
            device_name=device_name,
            gpu_model=gpu_model,
        )

        config.save({
            **config.load(),
            "device_token": result["device_token"],
            "device_id": str(result["device_id"]),
            "device_name": device_name,
            "gpu_model": gpu_model,
        })

        console.print("[green]✓[/green] Device authenticated successfully")
        console.print(f"  Device ID   : [bold]{result['device_id']}[/bold]")
        console.print(f"  Device name : {device_name}")
        console.print(f"  GPU         : {gpu_model or 'unknown'}")
        console.print(f"\n[dim]Config saved to ~/.paperclip/config.json[/dim]")
        console.print(f"\nNext step — set your GPU allocation:")
        console.print(f"  [bold]paperclip --allocate 70[/bold]")

    except Exception as e:
        console.print(f"[red]✗[/red] Authentication failed: {e}")
        sys.exit(1)


def _cmd_allocate(pct: int) -> None:
    token = _require_device_token()
    console.print(f"[dim]Setting allocation to {pct}%…[/dim]")
    try:
        result = api_client.set_allocation(token, pct)
        config.set_value("allocation_pct", pct)
        console.print(f"[green]✓[/green] GPU allocation set to [bold]{pct}%[/bold]")
        if pct == 0:
            console.print("  [dim]Job assignment is paused (0% allocation).[/dim]")
        elif pct == 100:
            console.print("  [dim]Full GPU available for rent.[/dim]")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to set allocation: {e}")
        sys.exit(1)


def _cmd_status() -> None:
    cfg = config.load()
    if not cfg.get("device_token"):
        console.print("[yellow]Not authenticated.[/yellow] Run [bold]paperclip --authenticate <token>[/bold]")
        return

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Device ID",   cfg.get("device_id", "—"))
    table.add_row("Device name", cfg.get("device_name", "—"))
    table.add_row("GPU",         cfg.get("gpu_model", "—"))
    table.add_row("Allocation",  f"{cfg.get('allocation_pct', '—')}%")
    table.add_row("API URL",     config.api_url())
    table.add_row("Mode",        "mock" if config.is_mock() else "real")
    table.add_row("Version",     __version__)
    console.print(table)


def _cmd_daemon() -> None:
    token = _require_device_token()
    gpu_model = config.get("gpu_model")
    mode = "mock" if config.is_mock() else "real"

    console.print(f"[bold]paperclip daemon[/bold] v{__version__} starting ({mode} mode)")
    console.print(f"  API    : {config.api_url()}")
    console.print(f"  Device : {config.get('device_id', '?')}")
    console.print("  Press Ctrl+C to stop.\n")

    heartbeat_interval = 60   # seconds between heartbeats
    poll_interval = 10        # seconds between job polls
    last_heartbeat = 0.0

    try:
        while True:
            now = time.time()

            # Heartbeat
            if now - last_heartbeat >= heartbeat_interval:
                try:
                    api_client.send_heartbeat(token, gpu_model=gpu_model)
                    logger.info("Heartbeat sent")
                    last_heartbeat = now
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")

            # Poll for job
            try:
                job = api_client.get_next_job(token)
            except Exception as e:
                logger.warning(f"Job poll failed: {e}")
                time.sleep(poll_interval)
                continue

            if job:
                console.print(f"\n[green]→[/green] Job received: [bold]{job['model_id']}[/bold] (id: {job['id'][:8]}…)")
                _execute_job(token, job)
            else:
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[dim]Daemon stopped.[/dim]")


def _execute_job(token: str, job: dict) -> None:
    job_id = job["id"]

    def progress_callback(progress: int, status: str) -> None:
        try:
            api_client.report_progress(token, job_id, progress, status)
            bar = "█" * (progress // 5) + "░" * (20 - progress // 5)
            console.print(f"  [{bar}] {progress}%", end="\r")
        except Exception as e:
            logger.warning(f"Progress report failed: {e}")

    try:
        run_job(job, progress_callback)
        console.print(f"\n[green]✓[/green] Job {job_id[:8]}… completed")
    except Exception as e:
        logger.error(f"Job failed: {e}")
        try:
            api_client.report_progress(token, job_id, 0, "failed", error_msg=str(e))
        except Exception:
            pass
        console.print(f"\n[red]✗[/red] Job {job_id[:8]}… failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_gpu() -> str | None:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return out.split("\n")[0] if out else None
    except Exception:
        return None


if __name__ == "__main__":
    cli()
