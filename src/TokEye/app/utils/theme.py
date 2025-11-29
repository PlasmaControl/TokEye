import gradio as gr


def make_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#f0f9ff",  # Very light sky blue
            c100="#e0f2fe",  # Light sky blue
            c200="#bae6fd",  # Soft sky blue
            c300="#7dd3fc",  # Light ocean blue
            c400="#38bdf8",  # Ocean blue
            c500="#0ea5e9",  # Deep ocean blue (primary)
            c600="#0284c7",  # Rich ocean blue
            c700="#0369a1",  # Dark ocean blue
            c800="#075985",  # Deep sea blue
            c900="#0c4a6e",  # Midnight ocean
            c950="#082f49",  # Deepest ocean
        ),
        secondary_hue=gr.themes.Color(
            c50="#fdf2f8",  # Lightest pink
            c100="#fce7f3",  # Very light pink
            c200="#fbcfe8",  # Light pink
            c300="#f9a8d4",  # Soft pink
            c400="#f472b6",  # Medium pink
            c500="#ec4899",  # Hot pink (secondary)
            c600="#db2777",  # Deep pink
            c700="#be185d",  # Rich pink
            c800="#9d174d",  # Dark pink
            c900="#831843",  # Very dark pink
            c950="#500724",  # Darkest pink
        ),
        neutral_hue=gr.themes.Color(
            c50="#fefefe",  # Almost white
            c100="#f8f8f6",  # Very light sand
            c200="#edeee0",  # Sandy background
            c300="#d4d5c8",  # Light taupe
            c400="#a5a696",  # Medium taupe
            c500="#76776a",  # Dark taupe
            c600="#5a5b4f",  # Darker taupe
            c700="#3f4037",  # Very dark gray
            c800="#2a2b24",  # Almost black
            c900="#1a1b16",  # Black
            c950="#0f100d",  # Pure dark
        ),
    ).set(
        # Text colors - dark for readability
        body_text_color="#1a1b16",
        body_text_color_dark="#1a1b16",
        body_text_color_subdued="#3f4037",
        body_text_color_subdued_dark="#3f4037",
        block_label_text_color="#075985",
        block_label_text_color_dark="#075985",
        block_title_text_color="#0369a1",
        block_title_text_color_dark="#0369a1",
        accordion_text_color="#1a1b16",
        accordion_text_color_dark="#1a1b16",
        table_text_color="#1a1b16",
        table_text_color_dark="#1a1b16",
        # Background colors - sandy beach tones
        body_background_fill="#edeee0",
        body_background_fill_dark="#edeee0",
        background_fill_primary="#edeee0",
        background_fill_primary_dark="#edeee0",
        background_fill_secondary="#d4d5c8",
        background_fill_secondary_dark="#d4d5c8",
        # Block/container styling - light and airy
        block_background_fill="#fefefe",
        block_background_fill_dark="#fefefe",
        block_border_color="#bae6fd",
        block_border_color_dark="#bae6fd",
        block_border_width="2px",
        block_label_background_fill="#e0f2fe",
        block_label_background_fill_dark="#e0f2fe",
        block_label_border_color="#7dd3fc",
        block_label_border_color_dark="#7dd3fc",
        block_label_border_width="1px",
        block_shadow="0 2px 8px 0 rgba(14, 165, 233, 0.15), 0 1px 3px -1px rgba(14, 165, 233, 0.2)",
        block_shadow_dark="0 2px 8px 0 rgba(14, 165, 233, 0.15), 0 1px 3px -1px rgba(14, 165, 233, 0.2)",
        block_title_background_fill="#bae6fd",
        block_title_background_fill_dark="#bae6fd",
        # Input styling - clean with light blue accents
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#7dd3fc",
        input_border_color_dark="#7dd3fc",
        input_border_color_focus="#0ea5e9",
        input_border_color_focus_dark="#0ea5e9",
        input_shadow="0 1px 3px 0 rgba(14, 165, 233, 0.1)",
        input_shadow_focus="0 0 0 3px rgba(56, 189, 248, 0.3)",
        # Button styling - vibrant ocean blue primary, pink secondary
        button_border_width="2px",
        button_primary_background_fill="#38bdf8",
        button_primary_background_fill_dark="#38bdf8",
        button_primary_background_fill_hover="#0ea5e9",
        button_primary_background_fill_hover_dark="#0ea5e9",
        button_primary_border_color="#0ea5e9",
        button_primary_border_color_dark="#0ea5e9",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#f472b6",
        button_secondary_background_fill_dark="#f472b6",
        button_secondary_background_fill_hover="#ec4899",
        button_secondary_background_fill_hover_dark="#ec4899",
        button_secondary_border_color="#db2777",
        button_secondary_border_color_dark="#db2777",
        button_secondary_border_color_hover="#be185d",
        button_secondary_text_color="#ffffff",
        button_secondary_text_color_dark="#ffffff",
        button_secondary_text_color_hover="#ffffff",
        button_cancel_background_fill="#f472b6",
        button_cancel_background_fill_hover="#ec4899",
        button_cancel_border_color="#db2777",
        button_cancel_text_color="#ffffff",
        # Checkbox styling
        checkbox_background_color="#ffffff",
        checkbox_background_color_dark="#ffffff",
        checkbox_label_background_fill="#ffffff",
        checkbox_label_background_fill_dark="#ffffff",
        checkbox_label_background_fill_selected="#38bdf8",
        checkbox_label_background_fill_selected_dark="#38bdf8",
        checkbox_label_border_color="#7dd3fc",
        checkbox_label_border_color_dark="#7dd3fc",
        checkbox_label_border_color_hover="#0ea5e9",
        checkbox_label_border_color_selected="#0ea5e9",
        checkbox_label_border_width="2px",
        checkbox_label_text_color_selected="#ffffff",
        checkbox_label_text_color_selected_dark="#ffffff",
        # Border radius - soft and rounded like beach pebbles
        button_large_radius="0.75rem",
        button_small_radius="0.5rem",
        # Slider styling - ocean blue
        slider_color="#38bdf8",
        slider_color_dark="#38bdf8",
        # Panel styling - light with blue borders
        panel_background_fill="#f8f8f6",
        panel_background_fill_dark="#f8f8f6",
        panel_border_color="#bae6fd",
        panel_border_color_dark="#bae6fd",
        # Table styling - alternating light rows
        table_even_background_fill="#fefefe",
        table_even_background_fill_dark="#fefefe",
        table_odd_background_fill="#f0f9ff",
        table_odd_background_fill_dark="#f0f9ff",
        table_border_color="#bae6fd",
        table_border_color_dark="#bae6fd",
        table_row_focus="#e0f2fe",
        table_row_focus_dark="#e0f2fe",
    )


theme = make_theme()
