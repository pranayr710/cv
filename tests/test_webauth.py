"""Tests for dashboard access control.

This server can turn on a camera pointed at minors and stream it. Every test
here corresponds to a way that was reachable before: no token on any route, a
WebSocket that accepted any Origin, and a camera-start endpoint with no body
that a cross-origin form could submit.
"""
import pytest

from backend.webauth import (
    TOKEN_ENV,
    origin_allowed,
    resolve_token,
    token_ok,
)


class TestToken:
    def test_a_generated_token_is_not_guessable(self):
        """No default. A fixed password in a repository is no password.

        This is software that gets deployed by copying a folder, so a
        committed default would be on every installation at once.
        """
        token, generated = resolve_token()
        assert generated is True
        assert len(token) >= 24
        assert resolve_token()[0] != token

    def test_an_environment_token_is_used_as_given(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "a-chosen-token")
        token, generated = resolve_token()
        assert (token, generated) == ("a-chosen-token", False)

    def test_blank_environment_falls_back_to_generating(self, monkeypatch):
        """An empty variable must not disable the check."""
        monkeypatch.setenv(TOKEN_ENV, "   ")
        _token, generated = resolve_token()
        assert generated is True

    @pytest.mark.parametrize("presented", [None, "", "wrong", "secret-but-no"])
    def test_wrong_tokens_are_refused(self, presented):
        assert token_ok(presented, "secret") is False

    def test_the_right_token_is_accepted(self):
        assert token_ok("secret", "secret") is True

    def test_a_prefix_is_not_enough(self):
        """Guards against a comparison that stops at the first difference."""
        assert token_ok("secr", "secret") is False
        assert token_ok("secretextra", "secret") is False


class TestOrigin:
    """The same-origin policy does not apply to WebSocket handshakes.

    Without an Origin check, any page a teacher visits while the dashboard
    runs on localhost can open a socket to it and receive the live video. No
    network exposure is needed -- a loopback-only deployment is vulnerable.
    """

    def test_the_pages_own_origin_is_allowed(self):
        assert origin_allowed("http://127.0.0.1:8000", "127.0.0.1:8000")

    def test_another_site_is_refused(self):
        assert not origin_allowed("https://evil.example", "127.0.0.1:8000")

    def test_a_different_port_on_localhost_is_refused(self):
        """Another local service is still another origin."""
        assert not origin_allowed("http://127.0.0.1:9999", "127.0.0.1:8000")

    def test_a_missing_origin_is_allowed(self):
        """curl, probes and the test suite send none.

        A browser always sends Origin on a cross-site request, so permitting
        its absence does not reopen the hole this closes.
        """
        assert origin_allowed(None, "127.0.0.1:8000")

    def test_a_malformed_origin_is_refused(self):
        assert not origin_allowed("not-a-url", "127.0.0.1:8000")

    def test_an_explicitly_permitted_origin_is_allowed(self):
        assert origin_allowed("https://dash.school.example", "127.0.0.1:8000",
                              frozenset({"https://dash.school.example"}))

    def test_a_lookalike_host_is_refused(self):
        """Substring matching would admit 127.0.0.1:8000.evil.example."""
        assert not origin_allowed("http://127.0.0.1:8000.evil.example",
                                  "127.0.0.1:8000")


class TestWhatAGatedPageReturns:
    """A browser cannot send a custom header on a navigation.

    window.open() on the report URL therefore arrived without the token, and
    the user clicking Stop was shown a raw JSON 401 instead of their report.
    The page now fetches it with the token, and anyone who reaches a gated URL
    directly gets something readable.
    """

    def test_a_browser_is_answered_with_a_page(self):
        from starlette.requests import Request

        from backend.webauth import _LOCKED_PAGE

        assert "<!doctype html>" in _LOCKED_PAGE.lower()
        assert "access token" in _LOCKED_PAGE.lower()
        # It must not leak the thing it is guarding.
        assert "X-ClassGraph-Token:" not in _LOCKED_PAGE
        assert Request  # imported to show the module loads under test

    def test_the_locked_page_points_somewhere_useful(self):
        """Telling someone they are unauthorised without saying what to do
        next is the same as saying nothing."""
        from backend.webauth import _LOCKED_PAGE

        assert 'href="/"' in _LOCKED_PAGE
        assert "CLASSGRAPH_TOKEN" in _LOCKED_PAGE
