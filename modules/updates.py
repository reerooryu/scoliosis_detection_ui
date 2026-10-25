# Update check: asks GitHub for the latest release of the app and compares
# its version with APP_VERSION. No Qt here, so it can be tested on its own.
#
# Releases are tagged "v" + version, e.g. "v0.2.0" (see README, "Make a
# release"). Draft and pre-release builds are not offered.

import re
from dataclasses import dataclass

import requests

from config import APP_VERSION, GITHUB_REPO

LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CHECK_TIMEOUT = 5  # seconds; the check must never hold anything up


@dataclass(frozen=True)
class UpdateInfo:
    version: str   # e.g. "0.2.0"
    url: str       # the release page, where the downloads are
    notes: str     # the release description, may be empty


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


def update_from_release(release, current=APP_VERSION):
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
