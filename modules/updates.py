# Update check: asks GitHub for the latest release of the app and compares
# its version with APP_VERSION. No Qt here, so it can be tested on its own.
# Installing an update is done by modules/updater.py.
#
# Releases are tagged "v" + version, e.g. "v0.2.0" (see README, "Make a
# release"). Draft and pre-release builds are not offered.

import re
import sys
from dataclasses import dataclass
from typing import Optional

import requests

from config import APP_VERSION, GITHUB_REPO

LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CHECK_TIMEOUT = 5  # seconds; the check must never hold anything up

# The release file for each kind of computer (.github/workflows/release.yml).
ASSET_NAMES = {
    "win32": "ScoliosisAI-Windows.zip",
    "darwin": "ScoliosisAI-Mac.zip",
}


@dataclass(frozen=True)
class UpdateAsset:
    """The release file for this computer."""
    name: str
    url: str               # direct download link
    size: int              # bytes
    sha256: Optional[str]  # lower-case hex from GitHub, or None if GitHub gave none


@dataclass(frozen=True)
class UpdateInfo:
    version: str                   # e.g. "0.2.0"
    url: str                       # the release page, where the downloads are
    notes: str                     # the release description, may be empty
    asset: Optional[UpdateAsset]   # None if the release has no file for this computer


def parse_version(text):
    """"v1.2.3" or "1.2" -> (1, 2, 3) / (1, 2, 0). None if not a version."""
    match = re.fullmatch(r"\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*", text or "")
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def is_newer(latest, current):
    """True if version text latest is above version text current."""
    latest_v, current_v = parse_version(latest), parse_version(current)
    return latest_v is not None and current_v is not None and latest_v > current_v


def asset_for_platform(release, platform=sys.platform):
    """The UpdateAsset for this kind of computer from a GitHub release dict,
    or None."""
    wanted = ASSET_NAMES.get(platform)
    for asset in release.get("assets") or []:
        if not isinstance(asset, dict) or asset.get("name") != wanted:
            continue
        url, size = asset.get("browser_download_url"), asset.get("size")
        if not url or not isinstance(size, int) or size <= 0:
            return None
        # GitHub gives a "digest" such as "sha256:3f2a..." for each file.
        digest = asset.get("digest") or ""
        sha256 = digest[len("sha256:"):].lower() if digest.startswith("sha256:") else None
        return UpdateAsset(name=wanted, url=url, size=size, sha256=sha256)
    return None


def update_from_release(release, current=APP_VERSION, platform=sys.platform):
    """An UpdateInfo if the GitHub release dict is newer than current, else None."""
    if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
        return None
    tag = release.get("tag_name") or ""
    if not is_newer(tag, current):
        return None
    return UpdateInfo(
        version=tag.lstrip("v"),
        url=release.get("html_url") or f"https://github.com/{GITHUB_REPO}/releases/latest",
        notes=(release.get("body") or "").strip(),
        asset=asset_for_platform(release, platform),
    )


def check_for_update(current=APP_VERSION, timeout=CHECK_TIMEOUT):
    """Ask GitHub for the latest release. Returns an UpdateInfo if it is
    newer, else None. Never raises: no internet, a blocked network or no
    release yet all just mean "no update"."""
    try:
        response = requests.get(
            LATEST_RELEASE_URL, timeout=timeout,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status_code != 200:
            return None
        return update_from_release(response.json(), current)
    except (requests.RequestException, ValueError):
        return None
