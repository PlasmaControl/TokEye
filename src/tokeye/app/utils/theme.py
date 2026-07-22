import gradio as gr

# Mirrors src/tokeye/gui/theme.py::COLORS on the diiid branch — keep in sync.
PALETTE = {
    "bg_window": "#13151a",  # page background
    "bg_surface": "#1b1e26",  # blocks/cards
    "bg_raised": "#22262f",  # secondary surfaces
    "bg_input": "#0f1115",  # inputs / plot canvas chrome
    "border": "#2a2f3a",
    "text": "#e9ecf1",
    "text_muted": "#8b93a1",
    "accent": "#45b8cb",  # cyan — avoids gist_heat + RdBu_r data colors
    "accent_hover": "#63d0e2",
    "accent_pressed": "#3aa2b3",
    "accent_text": "#08222a",  # text on accent-filled buttons
}

# footer{display:none !important} kept from the previous inline css.
CUSTOM_CSS = f"""
footer {{display: none !important}}
.logo-image img {{
    max-height: 120px;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}}
.tabs .tab-container button.selected {{
    color: {PALETTE["accent"]} !important;
    border-bottom: 2px solid {PALETTE["accent"]} !important;
}}
.block {{
    padding: 6px !important;
}}
textarea:disabled, input:disabled {{
    font-family: "JetBrains Mono", Consolas, monospace !important;
}}
"""


def make_theme() -> gr.themes.Base:
    """Dark control-room theme mirroring the native Qt GUI palette."""
    accent_hue = gr.themes.colors.Color(
        c50="#e6f7fa",
        c100="#c7edf3",
        c200="#9de1ea",
        c300="#7ad9e6",
        c400="#63d0e2",  # accent_hover
        c500="#45b8cb",  # accent
        c600="#3aa2b3",  # accent_pressed
        c700="#2f8794",
        c800="#256b75",
        c900="#1a4c53",
        c950="#0f2d31",
    )

    return gr.themes.Base(
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Cantarell", "sans-serif"],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "Consolas",
            "monospace",
        ],
        primary_hue=accent_hue,
    ).set(
        color_accent=PALETTE["accent"],
        color_accent_soft=PALETTE["accent"],
        color_accent_soft_dark=PALETTE["accent"],
        # Page background
        body_background_fill=PALETTE["bg_window"],
        body_background_fill_dark=PALETTE["bg_window"],
        background_fill_primary=PALETTE["bg_surface"],
        background_fill_primary_dark=PALETTE["bg_surface"],
        background_fill_secondary=PALETTE["bg_raised"],
        background_fill_secondary_dark=PALETTE["bg_raised"],
        # Text
        body_text_color=PALETTE["text"],
        body_text_color_dark=PALETTE["text"],
        body_text_color_subdued=PALETTE["text_muted"],
        body_text_color_subdued_dark=PALETTE["text_muted"],
        block_label_text_color=PALETTE["text_muted"],
        block_label_text_color_dark=PALETTE["text_muted"],
        block_title_text_color=PALETTE["text"],
        block_title_text_color_dark=PALETTE["text"],
        accordion_text_color=PALETTE["text"],
        accordion_text_color_dark=PALETTE["text"],
        table_text_color=PALETTE["text"],
        table_text_color_dark=PALETTE["text"],
        # Blocks/cards
        block_background_fill=PALETTE["bg_surface"],
        block_background_fill_dark=PALETTE["bg_surface"],
        block_border_color=PALETTE["border"],
        block_border_color_dark=PALETTE["border"],
        block_border_width="1px",
        block_shadow="none",
        block_shadow_dark="none",
        block_label_background_fill=PALETTE["bg_raised"],
        block_label_background_fill_dark=PALETTE["bg_raised"],
        block_label_border_color=PALETTE["border"],
        block_label_border_color_dark=PALETTE["border"],
        block_title_background_fill=PALETTE["bg_raised"],
        block_title_background_fill_dark=PALETTE["bg_raised"],
        border_color_primary=PALETTE["border"],
        border_color_primary_dark=PALETTE["border"],
        # Inputs
        input_background_fill=PALETTE["bg_input"],
        input_background_fill_dark=PALETTE["bg_input"],
        input_border_color=PALETTE["border"],
        input_border_color_dark=PALETTE["border"],
        input_border_color_focus=PALETTE["accent"],
        input_border_color_focus_dark=PALETTE["accent"],
        # Buttons — accent-filled primary
        button_primary_background_fill=PALETTE["accent"],
        button_primary_background_fill_dark=PALETTE["accent"],
        button_primary_background_fill_hover=PALETTE["accent_hover"],
        button_primary_background_fill_hover_dark=PALETTE["accent_hover"],
        button_primary_border_color=PALETTE["accent"],
        button_primary_border_color_dark=PALETTE["accent"],
        button_primary_text_color=PALETTE["accent_text"],
        button_primary_text_color_dark=PALETTE["accent_text"],
        button_secondary_background_fill=PALETTE["bg_raised"],
        button_secondary_background_fill_dark=PALETTE["bg_raised"],
        button_secondary_background_fill_hover=PALETTE["border"],
        button_secondary_background_fill_hover_dark=PALETTE["border"],
        button_secondary_border_color=PALETTE["border"],
        button_secondary_border_color_dark=PALETTE["border"],
        button_secondary_text_color=PALETTE["text"],
        button_secondary_text_color_dark=PALETTE["text"],
        # Slider
        slider_color=PALETTE["accent"],
        slider_color_dark=PALETTE["accent"],
        # Panels
        panel_background_fill=PALETTE["bg_surface"],
        panel_background_fill_dark=PALETTE["bg_surface"],
        panel_border_color=PALETTE["border"],
        panel_border_color_dark=PALETTE["border"],
        # Radius
        block_radius="8px",
        input_radius="6px",
        container_radius="8px",
        button_large_radius="6px",
        button_small_radius="6px",
    )


theme = make_theme()
