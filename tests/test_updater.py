# Tests for installing updates (modules/updater.py) with made-up app folders.
# The Mac-style swap runs on every computer; the real Windows PowerShell
# swap only on Windows (GitHub runs these tests on Windows and Mac too).

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from modules import updater
from modules.updates import UpdateAsset, UpdateInfo


def make_app(root, platform, marker):
    """A made-up app: the program file plus a marker file."""
    app = root / updater.APP_NAMES[platform]
    program = app / updater.PROGRAMS[platform]
    program.parent.mkdir(parents=True)
    program.write_text("program")
    (app / marker).write_text(marker)
    return app


def make_install(tmp_path, platform="darwin"):
    folder = tmp_path / "Applications"
    folder.mkdir()
    app = make_app(folder, platform, "old.txt")
    return updater.Install(platform, app, app / updater.PROGRAMS[platform], folder / updater.STAGING_NAME)


def release_zip(platform):
    """The bytes of a release zip holding a new made-up app."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        top = updater.APP_NAMES[platform]
        archive.writestr(f"{top}/{updater.PROGRAMS[platform].as_posix()}", "new program")
        archive.writestr(f"{top}/new.txt", "new")
    return buffer.getvalue()


class FakeResponse:
    """A streamed download. drop_after: raise a connection error after that
    many bytes, like a lost connection."""

    def __init__(self, data, status_code=200, headers=None, drop_after=None):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.drop_after = drop_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        sent = 0
        for start in range(0, len(self.data), size):
            chunk = self.data[start:start + size]
            if self.drop_after is not None and sent + len(chunk) > self.drop_after:
                yield chunk[:self.drop_after - sent]
                raise updater.requests.ConnectionError("connection lost")
            sent += len(chunk)
            yield chunk


def update_info(data, version="0.2.0", sha256="right", size=None):
    digest = hashlib.sha256(data).hexdigest() if sha256 == "right" else sha256
    asset = UpdateAsset("ScoliosisAI-Mac.zip", "https://example.org/m.zip", size or len(data), digest)
    return UpdateInfo(version, "https://example.org", "", asset)


@pytest.fixture
def downloads(monkeypatch):
    """Serve fake downloads. Set served["data"]; served["ranges"] = False
    makes the server ignore requests to continue; served["drop_after"] cuts
    the next download off after that many bytes. served["requests"] lists
    the Range header of each request (None for a full download)."""
    served = {"data": b"", "urls": [], "requests": [], "ranges": True, "drop_after": None}

    def fake_get(url, headers=None, **kwargs):
        served["urls"].append(url)
        wanted = (headers or {}).get("Range")
        served["requests"].append(wanted)
        drop, served["drop_after"] = served["drop_after"], None
        data = served["data"]
        if wanted and served["ranges"]:
            offset = int(wanted.split("=")[1].rstrip("-"))
            content_range = f"bytes {offset}-{len(data) - 1}/{len(data)}"
            return FakeResponse(data[offset:], 206, {"Content-Range": content_range}, drop)
        return FakeResponse(data, drop_after=drop)

    monkeypatch.setattr(updater.requests, "get", fake_get)
    return served


# --- where the app is -------------------------------------------------------

def test_current_install_layouts(tmp_path):
    assert updater.current_install(packaged=False) is None

    win_exe = tmp_path / "Scoliosis AI" / "ScoliosisAI" / "ScoliosisAI.exe"
    win_exe.parent.mkdir(parents=True)
    win = updater.current_install("win32", str(win_exe), packaged=True)
    assert win.app_path == win_exe.parent.resolve()
    assert win.staging_dir == win_exe.parent.parent.resolve() / updater.STAGING_NAME

    mac_exe = tmp_path / "Applications" / "ScoliosisAI.app" / "Contents" / "MacOS" / "ScoliosisAI"
    mac_exe.parent.mkdir(parents=True)
    mac = updater.current_install("darwin", str(mac_exe), packaged=True)
    assert mac.app_path.name == "ScoliosisAI.app"

    moved = tmp_path / "AppTranslocation" / "X" / "ScoliosisAI.app" / "Contents" / "MacOS" / "ScoliosisAI"
    moved.parent.mkdir(parents=True)
    assert updater.current_install("darwin", str(moved), packaged=True) is None
    assert updater.current_install("linux", str(win_exe), packaged=True) is None


# --- download and unpack ----------------------------------------------------

def test_prepare_update_stages_once(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    info = update_info(downloads["data"])

    app = updater.prepare_update(info, install)
    assert (app / "new.txt").read_text() == "new"
    assert json.loads((install.staging_dir / updater.READY_FILE).read_text())["version"] == "0.2.0"
    assert not list(install.staging_dir.glob("*.zip*"))       # the download is removed

    updater.prepare_update(info, install)                     # already staged: no new download
    assert len(downloads["urls"]) == 1


@pytest.mark.parametrize("problem", ["checksum", "size"])
def test_damaged_download_is_refused(tmp_path, downloads, problem):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    if problem == "checksum":
        info = update_info(downloads["data"], sha256="0" * 64)
    else:
        info = update_info(downloads["data"], size=len(downloads["data"]) + 1)
    with pytest.raises(updater.UpdateError):
        updater.prepare_update(info, install)
    assert updater._read_ready(install.staging_dir) is None
    assert updater.apply_pending_update(install, current="0.1.0", launch=False) is False
    assert (install.app_path / "old.txt").exists()


# --- continuing a download ---------------------------------------------------

def part_file(install, info):
    return install.staging_dir / updater._part_path(info)


def test_cut_off_download_continues(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    info = update_info(downloads["data"])
    half = len(downloads["data"]) // 2

    downloads["drop_after"] = half
    with pytest.raises(updater.DownloadPaused):
        updater.prepare_update(info, install)
    assert part_file(install, info).stat().st_size == half     # kept

    # The next start keeps the partial download, then continues it.
    assert updater.apply_pending_update(install, current="0.1.0", launch=False) is False
    assert part_file(install, info).exists()
    seen = []
    app = updater.prepare_update(info, install, progress=lambda done, total: seen.append((done, total)))
    assert downloads["requests"] == [None, f"bytes={half}-"]
    assert (app / "new.txt").read_text() == "new"
    assert seen[0][0] == half and seen[-1] == (len(downloads["data"]), len(downloads["data"]))


def test_server_ignoring_continue_starts_again(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    info = update_info(downloads["data"])
    downloads["drop_after"] = 100
    with pytest.raises(updater.DownloadPaused):
        updater.prepare_update(info, install)
    downloads["ranges"] = False
    app = updater.prepare_update(info, install)
    assert (app / "new.txt").exists()


def test_damaged_partial_download_is_thrown_away(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    info = update_info(downloads["data"])
    install.staging_dir.mkdir()
    part_file(install, info).write_bytes(b"x" * 100)            # wrong bytes
    with pytest.raises(updater.UpdateError) as caught:
        updater.prepare_update(info, install)
    assert not isinstance(caught.value, updater.DownloadPaused)
    assert not part_file(install, info).exists()
    app = updater.prepare_update(info, install)                 # starts again, works
    assert downloads["requests"][-1] is None and (app / "new.txt").exists()


def test_no_checksum_means_no_continuing(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    info = update_info(downloads["data"], sha256=None)
    install.staging_dir.mkdir()
    part_file(install, info).write_bytes(downloads["data"][:50])
    app = updater.prepare_update(info, install)
    assert downloads["requests"] == [None] and (app / "new.txt").exists()


def test_cleanup_drops_partial_download_of_an_older_version(tmp_path):
    install = make_install(tmp_path)
    install.staging_dir.mkdir()
    old = install.staging_dir / "0.1.0-ScoliosisAI-Mac.zip.part"
    newer = install.staging_dir / "0.3.0-ScoliosisAI-Mac.zip.part"
    old.write_bytes(b"1")
    newer.write_bytes(b"2")
    updater.cleanup(install, current="0.2.0")
    assert newer.exists() and not old.exists()
    updater.cleanup(install, current="0.3.0")
    assert not install.staging_dir.exists()


# --- swap -------------------------------------------------------------------

def test_mac_swap_and_cleanup(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    updater.prepare_update(update_info(downloads["data"]), install)

    assert updater.apply_pending_update(install, current="0.1.0", launch=False) is True
    assert (install.app_path / "new.txt").exists()
    assert not (install.app_path / "old.txt").exists()

    # The new version's first start removes the leftovers.
    assert updater.apply_pending_update(install, current="0.2.0", launch=False) is False
    assert not install.staging_dir.exists()
    assert not updater._old_app_path(install).exists()
    assert (install.app_path / "new.txt").exists()


def test_not_newer_is_not_installed(tmp_path, downloads):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    updater.prepare_update(update_info(downloads["data"], version="0.1.0"), install)
    assert updater.apply_pending_update(install, current="0.1.0", launch=False) is False
    assert (install.app_path / "old.txt").exists()
    assert not install.staging_dir.exists()


def test_failed_update_is_dropped_after_two_tries(tmp_path, downloads, monkeypatch):
    install = make_install(tmp_path)
    downloads["data"] = release_zip("darwin")
    updater.prepare_update(update_info(downloads["data"]), install)

    def broken_swap(install, staged):
        raise OSError("locked")

    monkeypatch.setattr(updater, "_swap_mac", broken_swap)
    for _ in range(updater.MAX_ATTEMPTS):
        assert updater.install_pending_update(install) is False   # never raises
    assert updater.install_pending_update(install) is False
    assert not install.staging_dir.exists()                       # given up
    assert (install.app_path / "old.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell swap")
def test_windows_swap_script(tmp_path):
    # A real swap with Windows PowerShell. The "app" is a copy of whoami.exe,
    # so starting it afterwards just prints a name and exits.
    whoami = Path(os.environ["SystemRoot"]) / "System32" / "whoami.exe"
    folder = tmp_path / "Scoliosis AI"
    folder.mkdir()
    app = make_app(folder, "win32", "old.txt")
    shutil.copy(whoami, app / "ScoliosisAI.exe")
    install = updater.Install("win32", app, app / "ScoliosisAI.exe", folder / updater.STAGING_NAME)
    staged = make_app(install.staging_dir / "0.2.0", "win32", "new.txt")
    shutil.copy(whoami, staged / "ScoliosisAI.exe")

    # The "old app" is a short-lived process the script must wait for.
    old_app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    started = time.monotonic()
    helper = updater._start_windows_swap(install, staged, old_app.pid)
    assert helper.wait(timeout=120) == 0
    assert time.monotonic() - started >= 1.5           # it waited for the old app
    assert (app / "new.txt").exists()
    assert not (app / "old.txt").exists()
    assert not updater._old_app_path(install).exists()
    assert not staged.exists()
