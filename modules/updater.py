# Installs a new version of the packaged app by itself (Windows and Mac).
#
#   1. modules/updates.py finds a newer release on GitHub.
#   2. prepare_update() downloads the zip for this computer in the background,
#      checks its size and checksum, and unpacks it into a staging folder next
#      to the app (".ScoliosisAI-update"). "ready.json" is written last, so a
#      half-finished download is never installed.
#   3. install_pending_update() swaps the new version in, at the next start or
#      after "Update Now" closes the window:
#        Windows: a running program can't replace its own folder, so a small
#          PowerShell script waits for the app to close, swaps the folders and
#          starts the new version.
#        Mac: the app bundle is swapped directly and the new one opened.
#      If anything fails, the current version keeps running; a staged update
#      that fails twice is thrown away.
#
# Running from source, none of this is used (current_install() returns None)
# and the app only offers the download page. The model file and the settings
# are never touched.

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from config import APP_VERSION
from modules.app_paths import LOG_PATH, is_packaged
from modules.updates import is_newer

logger = logging.getLogger(__name__)

STAGING_NAME = ".ScoliosisAI-update"
READY_FILE = "ready.json"
HELPER_NAME = "apply_update.ps1"
MAX_ATTEMPTS = 2              # a staged update that failed to install this often is dropped
DOWNLOAD_TIMEOUT = (10, 60)   # seconds: to connect, and between two pieces of data
CHUNK_SIZE = 1024 * 1024

# Top-level name inside each release zip, and the program inside it.
APP_NAMES = {"win32": "ScoliosisAI", "darwin": "ScoliosisAI.app"}
PROGRAMS = {"win32": Path("ScoliosisAI.exe"), "darwin": Path("Contents", "MacOS", "ScoliosisAI")}


class UpdateError(Exception):
    """A download or install step failed. The app keeps its current version."""


@dataclass(frozen=True)
class Install:
    """Where the running app is."""
    platform: str       # "win32" or "darwin"
    app_path: Path      # Windows: the ScoliosisAI folder. Mac: ScoliosisAI.app
    executable: Path    # the program started after a Windows swap
    staging_dir: Path   # next to app_path, so the swap is a quick rename


def current_install(platform=sys.platform, executable=None, packaged=None):
    """The running app's Install, or None if it can't update itself: run
    from source, not Windows or Mac, its folder is read-only, or macOS runs
    it from a temporary read-only copy (opened from Downloads instead of
    Applications)."""
    packaged = is_packaged() if packaged is None else packaged
    if not packaged:
        return None
    exe = Path(executable or sys.executable).resolve()
    if platform == "win32":
        app = exe.parent
    elif platform == "darwin":
        app = exe.parents[2]          # ScoliosisAI.app/Contents/MacOS/ScoliosisAI
        if app.suffix != ".app" or "AppTranslocation" in app.parts:
            return None
    else:
        return None
    if not os.access(app.parent, os.W_OK):
        return None
    return Install(platform, app, exe, app.parent / STAGING_NAME)


def staged_app(install, version):
    """Where the unpacked new version of the app is staged."""
    return install.staging_dir / version / APP_NAMES[install.platform]


def _has_program(app, platform):
    return (app / PROGRAMS[platform]).is_file()


# ---------------------------------------------------------------------------
# Step 2: download and unpack (background thread)
# ---------------------------------------------------------------------------

def prepare_update(info, install, cancel=None):
    """Download and unpack info's release file (an updates.UpdateInfo) next
    to the app. Returns the staged app's path. A version that is already
    staged is reused, not downloaded again. cancel is an optional
    threading.Event. Raises UpdateError."""
    if info.asset is None:
        raise UpdateError("The release has no file for this computer.")
    app = staged_app(install, info.version)
    ready = _read_ready(install.staging_dir)
    if ready and ready.get("version") == info.version and _has_program(app, install.platform):
        return app

    _remove(install.staging_dir)
    try:
        install.staging_dir.mkdir(parents=True)
    except OSError as exc:
        raise UpdateError(f"Could not create {install.staging_dir}: {exc}") from exc
    _hide(install.staging_dir)

    zip_path = install.staging_dir / info.asset.name
    _download(info.asset, zip_path, cancel)
    _unpack(zip_path, install.staging_dir / info.version, install.platform)
    zip_path.unlink()
    if not _has_program(app, install.platform):
        raise UpdateError(f"The downloaded file does not contain {APP_NAMES[install.platform]}.")
    _write_ready(install.staging_dir, {"version": info.version, "attempts": 0})
    logger.info("Update %s is ready to install", info.version)
    return app


def _download(asset, path, cancel):
    """Save asset (an updates.UpdateAsset) to path, checking its size and,
    if GitHub gave one, its SHA-256 checksum."""
    part = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with requests.get(asset.url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with open(part, "wb") as out:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if cancel is not None and cancel.is_set():
                        raise UpdateError("Download cancelled.")
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size != asset.size:
            raise UpdateError(f"Download incomplete: {size} of {asset.size} bytes.")
        if asset.sha256 and digest.hexdigest() != asset.sha256:
            raise UpdateError("The downloaded file is damaged (checksum does not match).")
        os.replace(part, path)
    except requests.RequestException as exc:
        raise UpdateError(f"Download failed: {exc}") from exc
    except OSError as exc:
        raise UpdateError(f"Could not save the download: {exc}") from exc
    finally:
        if part.exists():
            part.unlink()


def _unpack(zip_path, dest, platform):
    try:
        dest.mkdir(parents=True)
        if platform == "darwin" and sys.platform == "darwin":
            # ditto keeps the app bundle's links, permissions and attributes,
            # which Python's zipfile does not.
            subprocess.run(["ditto", "-x", "-k", str(zip_path), str(dest)], check=True, capture_output=True)
        else:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(dest)
    except (OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        raise UpdateError(f"Could not unpack the update: {exc}") from exc


# ---------------------------------------------------------------------------
# Step 3: swap in the new version
# ---------------------------------------------------------------------------

def install_pending_update(install=None):
    """Install a staged newer version, if there is one. Returns True if the
    new version is being started (the caller must exit at once), False to
    carry on with this version. Never raises."""
    try:
        install = install or current_install()
        if install is None:
            return False
        return apply_pending_update(install)
    except Exception:
        logger.exception("Installing the update failed; keeping the current version")
        return False


def apply_pending_update(install, current=APP_VERSION, pid=None, launch=True):
    """The work of install_pending_update(). launch=False skips opening the
    new version on a Mac (for tests)."""
    ready = _read_ready(install.staging_dir)
    if ready is None or not is_newer(ready["version"], current):
        cleanup(install)
        return False

    app = staged_app(install, ready["version"])
    attempts = ready.get("attempts", 0) if isinstance(ready.get("attempts"), int) else MAX_ATTEMPTS
    if attempts >= MAX_ATTEMPTS or not _has_program(app, install.platform):
        logger.warning("Dropping staged update %s (attempts: %s)", ready["version"], attempts)
        cleanup(install)
        return False

    ready["attempts"] = attempts + 1
    _write_ready(install.staging_dir, ready)
    logger.info("Installing update %s", ready["version"])
    if install.platform == "win32":
        _start_windows_swap(install, app, pid or os.getpid())
    else:
        _swap_mac(install, app)
        if launch:
            subprocess.Popen(["open", "-n", str(install.app_path)])
    return True


def cleanup(install):
    """Remove the staging folder and the old version left by a swap."""
    _remove(install.staging_dir)
    _remove(_old_app_path(install))


def _old_app_path(install):
    if install.platform == "win32":
        return install.app_path.with_name(install.app_path.name + ".old")
    return install.app_path.with_name("." + install.app_path.name + ".old")


def _swap_mac(install, staged):
    """Replace the app bundle with the staged one. A running app's bundle can
    be renamed on a Mac; the old one is removed at the next start."""
    old = _old_app_path(install)
    _remove(old)
    os.rename(install.app_path, old)
    try:
        shutil.move(str(staged), str(install.app_path))
    except OSError:
        if not install.app_path.exists():
            os.rename(old, install.app_path)   # put the current version back
        raise


# Run by Windows PowerShell 5.1 (part of Windows 10 and 11). ASCII only.
HELPER_SCRIPT = r"""# Installs a downloaded update of Scoliosis AI (written by modules/updater.py).
# Waits for the old app to close, swaps the app folder for the new one, then
# starts the app again. If the swap fails, the old version is kept and started.
param(
    [int]$ProcId,
    [string]$Target,
    [string]$Staged,
    [string]$Exe,
    [string]$LogPath
)
if ($LogPath) { try { Start-Transcript -LiteralPath $LogPath -Force | Out-Null } catch { } }
$old = "$Target.old"
$oldName = Split-Path -Leaf $old
$targetName = Split-Path -Leaf $Target

# 1. Wait for the old app to close (at most two minutes).
try { Wait-Process -Id $ProcId -Timeout 120 -ErrorAction Stop } catch { }

# 2. Swap the folders. Files can stay locked for a moment after the app
#    closes (or while antivirus scans them), so keep trying for 20 seconds.
$swapped = $false
for ($i = 0; ($i -lt 40) -and (-not $swapped); $i++) {
    try {
        if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction Stop }
        Rename-Item -LiteralPath $Target -NewName $oldName -ErrorAction Stop
        try {
            Move-Item -LiteralPath $Staged -Destination $Target -ErrorAction Stop
            $swapped = $true
        } catch {
            Rename-Item -LiteralPath $old -NewName $targetName -ErrorAction Stop
            throw
        }
    } catch {
        Write-Output "Attempt $i failed: $_"
        Start-Sleep -Milliseconds 500
    }
}
# Never leave the app missing.
if ((-not (Test-Path -LiteralPath $Target)) -and (Test-Path -LiteralPath $old)) {
    Rename-Item -LiteralPath $old -NewName $targetName
}
Write-Output "Swapped: $swapped"

# 3. Start the app: the new version if the swap worked, else the old one.
Start-Process -FilePath $Exe -WorkingDirectory $Target
if ($swapped) { try { Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction Stop } catch { } }
if ($LogPath) { try { Stop-Transcript | Out-Null } catch { } }
"""


def _start_windows_swap(install, staged, pid, shell="powershell.exe"):
    """Start the PowerShell helper that swaps in the staged version once
    process pid has closed. Returns the helper's Popen (tests wait on it)."""
    script = install.staging_dir / HELPER_NAME
    script.write_text(HELPER_SCRIPT, encoding="utf-8-sig")
    command = [
        shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
        "-ProcId", str(pid),
        "-Target", str(install.app_path),
        "-Staged", str(staged),
        "-Exe", str(install.executable),
        "-LogPath", str(LOG_PATH.with_name("update-log.txt")),
    ]
    flags = 0
    if sys.platform == "win32":
        # No console window, and not closed together with this app.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        command, creationflags=flags, close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read_ready(staging_dir):
    try:
        data = json.loads((staging_dir / READY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("version"), str) else None


def _write_ready(staging_dir, data):
    temp = staging_dir / (READY_FILE + ".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temp, staging_dir / READY_FILE)


def _remove(path):
    """Delete a file or folder if it exists. Failures are only logged."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as exc:
        logger.warning("Could not remove %s: %s", path, exc)


def _hide(path):
    """Hide the staging folder in Windows Explorer (Finder hides dot folders)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x02)   # FILE_ATTRIBUTE_HIDDEN
