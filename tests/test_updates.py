from modules.updates import asset_for_platform, is_newer, parse_version, update_from_release

RELEASE_ASSETS = [
    {"name": "ScoliosisAI-Windows.zip", "browser_download_url": "https://example.org/w.zip",
     "size": 100, "digest": "sha256:ABC123"},
    {"name": "ScoliosisAI-Mac.zip", "browser_download_url": "https://example.org/m.zip", "size": 200},
]


def test_parse_version():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("0.2") == (0, 2, 0)
    assert parse_version("3") == (3, 0, 0)
    assert parse_version("v1.2.3-beta") is None
    assert parse_version("") is None
    assert parse_version(None) is None


def test_is_newer_compares_numbers_not_text():
    assert is_newer("v0.10.0", "0.9.0")
    assert is_newer("v0.1.1", "0.1.0")
    assert not is_newer("v0.1.0", "0.1.0")
    assert not is_newer("v0.0.9", "0.1.0")
    assert not is_newer("nightly", "0.1.0")


def test_update_from_release():
    release = {"tag_name": "v0.2.0", "html_url": "https://example.org/r", "body": " notes "}
    info = update_from_release(release, current="0.1.0")
    assert info.version == "0.2.0"
    assert info.url == "https://example.org/r"
    assert info.notes == "notes"


def test_asset_for_each_computer():
    release = {"assets": RELEASE_ASSETS}
    windows = asset_for_platform(release, "win32")
    assert (windows.name, windows.url, windows.size, windows.sha256) == (
        "ScoliosisAI-Windows.zip", "https://example.org/w.zip", 100, "abc123")
    mac = asset_for_platform(release, "darwin")
    assert mac.size == 200 and mac.sha256 is None
    assert asset_for_platform(release, "linux") is None
    assert asset_for_platform({}, "win32") is None


def test_update_carries_the_asset():
    release = {"tag_name": "v0.2.0", "assets": RELEASE_ASSETS}
    assert update_from_release(release, current="0.1.0", platform="darwin").asset.name == "ScoliosisAI-Mac.zip"
    assert update_from_release(release, current="0.1.0", platform="linux").asset is None


def test_no_update_for_same_draft_or_prerelease():
    assert update_from_release({"tag_name": "v0.1.0"}, current="0.1.0") is None
    assert update_from_release({"tag_name": "v0.2.0", "draft": True}, current="0.1.0") is None
    assert update_from_release({"tag_name": "v0.2.0", "prerelease": True}, current="0.1.0") is None
    assert update_from_release({}, current="0.1.0") is None
    assert update_from_release(None, current="0.1.0") is None
