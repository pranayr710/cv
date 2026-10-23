"""The dashboard's static checks, and proof that they fail when they should.

A checker nobody has seen fail is a checker nobody knows works. Each class
here feeds it the defect it was written for and asserts it is caught, then
feeds it correct code and asserts it is quiet -- a check that fires on
everything is as useless as one that fires on nothing, and noisier.

All three defects share a symptom: a dead button or a missing panel, with no
error unless a console is open. None is visible in a diff.
"""
from pathlib import Path

from tools.check_js_strings import (
    broken_strings,
    code_only,
    missing_handlers,
    undefined_calls,
)

APP = Path("tools/static/app.js")
HTML = Path("tools/static/index.html")


class TestAStringSpanningALine:
    def test_it_is_caught(self):
        assert broken_strings('const a = "one\ntwo";') == [1]

    def test_a_template_literal_may_span_lines(self):
        assert broken_strings("const a = `one\ntwo`;") == []

    def test_an_apostrophe_in_a_block_comment_is_not_a_string(self):
        """The false positive this checker produced against itself, once."""
        assert broken_strings("/* one's thing */\nconst a = 1;") == []

    def test_an_escaped_quote_does_not_end_the_string(self):
        assert broken_strings('const a = "say \\"hi\\"";') == []


class TestABlankedLiteral:
    def test_code_survives(self):
        assert "const a" in code_only('const a = "text";')

    def test_string_contents_do_not(self):
        assert "rgba" not in code_only('const a = "rgba(1,2,3,.5)";')

    def test_line_numbers_are_preserved(self):
        source = 'const a = `one\ntwo`;\nconst b = 2;'
        assert code_only(source).count("\n") == source.count("\n")


class TestAnInlineHandlerWithNoFunction:
    def test_it_is_caught(self):
        html = '<button onclick="gone()">x</button>'
        assert missing_handlers(html, "function here() {}") == ["gone"]

    def test_a_defined_handler_is_quiet(self):
        html = '<button onclick="here()">x</button>'
        assert missing_handlers(html, "function here() {}") == []

    def test_a_const_arrow_handler_counts_as_defined(self):
        html = '<button onclick="here()">x</button>'
        assert missing_handlers(html, "const here = () => 1;") == []


class TestACallToADeletedFunction:
    """This happened twice in one edit, and the second one was found by this
    check rather than by anybody reading the file."""

    def test_it_is_caught(self):
        assert undefined_calls("function a() { gone(); }") == ["gone"]

    def test_a_defined_call_is_quiet(self):
        assert undefined_calls("function a() { b(); }\nfunction b() {}") == []

    def test_a_method_call_is_not_reported(self):
        """`a.b()` belongs to an object this cannot see. Guessing at those
        would bury the real finding in noise."""
        assert undefined_calls("function a(x) { x.whatever(); }") == []

    def test_a_name_inside_a_string_is_not_a_call(self):
        """SVG attributes are full of translate(...) and rgba(...)."""
        source = 'function a() { return "translate(1,2) rgba(0,0,0,.5)"; }'
        assert undefined_calls(source) == []

    def test_a_browser_constructor_is_not_reported(self):
        assert undefined_calls("function a() { return new Set(); }") == []


class TestTheRealDashboard:
    """The files these exist to protect. If any of this fails the dashboard
    is broken right now, not at some future edit."""

    def test_no_string_spans_a_line(self):
        assert broken_strings(APP.read_text(encoding="utf-8")) == []

    def test_every_inline_handler_exists(self):
        assert missing_handlers(HTML.read_text(encoding="utf-8"),
                                APP.read_text(encoding="utf-8")) == []

    def test_every_bare_call_resolves(self):
        assert undefined_calls(APP.read_text(encoding="utf-8")) == []

    def test_the_renderers_the_mode_switch_names_all_exist(self):
        """setGraphMode dispatches by string, so a renamed renderer is not a
        reference error anywhere -- the tab just silently draws nothing."""
        source = APP.read_text(encoding="utf-8")
        for name in ("renderRoomMap", "renderObjectLinks",
                     "renderScoreCompare", "renderInteractionMatrix"):
            assert f"function {name}(" in source, name
