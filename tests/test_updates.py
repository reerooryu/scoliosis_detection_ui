from modules.updates import is_newer, parse_version, update_from_release


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


def test_no_update_for_same_draft_or_prerelease():
    assert update_from_release({"tag_name": "v0.1.0"}, current="0.1.0") is None
    assert update_from_release({"tag_name": "v0.2.0", "draft": True}, current="0.1.0") is None
    assert update_from_release({"tag_name": "v0.2.0", "prerelease": True}, current="0.1.0") is None
    assert update_from_release({}, current="0.1.0") is None
    assert update_from_release(None, current="0.1.0") is None
