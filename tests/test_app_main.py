"""Tests for src/tokeye/app/__main__.py main() function."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tokeye.app.__main__ import DEFAULT_PORT, MAX_PORT_ATTEMPTS, create_app, main
from tokeye.app.utils.theme import PALETTE, make_theme


class TestMainPortRetry:
    """Test the port-retry logic in main()."""

    def test_retry_upward_on_oserror(self):
        """Port should increment on OSError, not decrement."""
        fake_app = Mock()
        # Fail twice on ports 7860 and 7861, succeed on 7862
        fake_app.launch.side_effect = [
            OSError("Port in use"),
            OSError("Port in use"),
            None,  # success
        ]

        with patch("tokeye.app.__main__.create_app", return_value=fake_app):
            main(port=DEFAULT_PORT)

        # Verify the ports tried were 7860, 7861, 7862
        assert fake_app.launch.call_count == 3
        calls = fake_app.launch.call_args_list
        assert calls[0][1]["server_port"] == DEFAULT_PORT
        assert calls[1][1]["server_port"] == DEFAULT_PORT + 1
        assert calls[2][1]["server_port"] == DEFAULT_PORT + 2

    def test_return_on_success(self):
        """main() should return (not fall through) after successful launch."""
        fake_app = Mock()
        fake_app.launch.return_value = None  # success

        with patch("tokeye.app.__main__.create_app", return_value=fake_app):
            result = main(port=DEFAULT_PORT)

        # Should return cleanly
        assert result is None
        # Should only try once
        assert fake_app.launch.call_count == 1

    def test_systemeexit_after_all_attempts_fail(self):
        """Should raise SystemExit naming the port range after all attempts fail."""
        fake_app = Mock()
        fake_app.launch.side_effect = OSError("Port in use")

        with patch("tokeye.app.__main__.create_app", return_value=fake_app), \
                pytest.raises(SystemExit) as exc_info:
            main(port=DEFAULT_PORT)

        # Check the message names the port range
        message = str(exc_info.value)
        assert "7860" in message
        assert str(DEFAULT_PORT + MAX_PORT_ATTEMPTS - 1) in message
        # Should have tried all MAX_PORT_ATTEMPTS
        assert fake_app.launch.call_count == MAX_PORT_ATTEMPTS

    def test_launch_kwargs_preserved(self):
        """launch() should receive all original kwargs, only port changes."""
        fake_app = Mock()
        fake_app.launch.return_value = None

        with patch("tokeye.app.__main__.create_app", return_value=fake_app):
            main(port=7777, share=True, open_browser=True)

        fake_app.launch.assert_called_once_with(
            share=True, inbrowser=True, server_port=7777
        )

    def test_non_oserror_exceptions_propagate(self):
        """Non-OSError exceptions should propagate (not be caught by retry logic)."""
        fake_app = Mock()
        fake_app.launch.side_effect = RuntimeError("Some other error")

        with patch("tokeye.app.__main__.create_app", return_value=fake_app), \
                pytest.raises(RuntimeError):
            main(port=DEFAULT_PORT)


class TestDarkControlRoomTheme:
    """Tests for the dark control-room theme (mirrors the native Qt GUI palette)."""

    def test_body_background_is_dark_in_both_variants(self):
        """The page background must be forced dark regardless of browser preference."""
        t = make_theme()
        assert t.body_background_fill == PALETTE["bg_window"]
        assert t.body_background_fill_dark == PALETTE["bg_window"]
        assert t.body_background_fill == t.body_background_fill_dark

    def test_block_background_is_dark_in_both_variants(self):
        t = make_theme()
        assert t.block_background_fill == PALETTE["bg_surface"]
        assert t.block_background_fill_dark == PALETTE["bg_surface"]

    def test_body_text_color_is_dark_in_both_variants(self):
        t = make_theme()
        assert t.body_text_color == PALETTE["text"]
        assert t.body_text_color_dark == PALETTE["text"]

    def test_primary_button_uses_accent_in_both_variants(self):
        """The accent-filled primary button must match the shared palette exactly."""
        t = make_theme()
        assert t.button_primary_background_fill == PALETTE["accent"]
        assert t.button_primary_background_fill_dark == PALETTE["accent"]
        assert t.button_primary_background_fill == t.button_primary_background_fill_dark

    def test_primary_button_text_uses_accent_text_in_both_variants(self):
        t = make_theme()
        assert t.button_primary_text_color == PALETTE["accent_text"]
        assert t.button_primary_text_color_dark == PALETTE["accent_text"]

    def test_input_background_uses_bg_input_in_both_variants(self):
        t = make_theme()
        assert t.input_background_fill == PALETTE["bg_input"]
        assert t.input_background_fill_dark == PALETTE["bg_input"]

    def test_input_focus_border_uses_accent_in_both_variants(self):
        t = make_theme()
        assert t.input_border_color_focus == PALETTE["accent"]
        assert t.input_border_color_focus_dark == PALETTE["accent"]

    def test_slider_uses_accent_in_both_variants(self):
        t = make_theme()
        assert t.slider_color == PALETTE["accent"]
        assert t.slider_color_dark == PALETTE["accent"]

    def test_block_label_text_uses_muted_color_in_both_variants(self):
        t = make_theme()
        assert t.block_label_text_color == PALETTE["text_muted"]
        assert t.block_label_text_color_dark == PALETTE["text_muted"]

    def test_palette_hex_values_are_the_cross_branch_contract(self):
        """These hex values mirror gui/theme.py::COLORS on the diiid branch —
        change only in lockstep with that file."""
        assert PALETTE == {
            "bg_window": "#13151a",
            "bg_surface": "#1b1e26",
            "bg_raised": "#22262f",
            "bg_input": "#0f1115",
            "border": "#2a2f3a",
            "text": "#e9ecf1",
            "text_muted": "#8b93a1",
            "accent": "#45b8cb",
            "accent_hover": "#63d0e2",
            "accent_pressed": "#3aa2b3",
            "accent_text": "#08222a",
        }

    def test_create_app_builds_with_dark_theme(self):
        """create_app() must still build cleanly with the new theme + CSS wired in."""
        app = create_app()
        assert app is not None
