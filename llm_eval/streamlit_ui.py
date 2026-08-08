from __future__ import annotations

import os

import streamlit as st


_PLACEHOLDER_API_KEYS = {
    "your-api-key",
    "your-openrouter-api-key",
    "replace-with-your-api-key",
}


def apply_ui_style() -> None:
    """Apply a small shared visual layer without overriding the active theme."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1380px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.8rem;
            padding: 0.8rem 1rem;
        }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.45rem, 2.2vw, 2rem);
        }
        [data-testid="stAlert"] {
            border-radius: 0.8rem;
        }
        div.stButton > button {
            border-radius: 0.65rem;
            min-height: 2.75rem;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        .evaluation-nav {
            display: grid;
            gap: 0.35rem;
            margin: 0.45rem 0 1.4rem;
        }
        .evaluation-nav a {
            color: inherit;
            text-decoration: none;
            padding: 0.62rem 0.75rem;
            border-radius: 0.55rem;
        }
        .evaluation-nav a:hover {
            background: rgba(128, 128, 128, 0.12);
        }
        .evaluation-nav a.active {
            background: rgba(128, 128, 128, 0.22);
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(active_page: str) -> None:
    links = (
        ("overview", "/", "Overview & previous benchmark"),
        ("run", "/Run_Evaluation", "Run evaluation"),
        ("results", "/Framework_Results", "Framework results"),
        ("comparison", "/Prompt_Comparison", "Prompt comparison"),
    )
    items = "".join(
        f'<a class="{"active" if page == active_page else ""}" href="{href}" target="_self">{label}</a>'
        for page, href, label in links
    )
    st.sidebar.markdown("### LLM Evaluation")
    st.sidebar.caption("Configure · Run · Analyze · Compare")
    st.sidebar.markdown(f'<nav class="evaluation-nav">{items}</nav>', unsafe_allow_html=True)


def render_workflow(active_step: int) -> None:
    labels = (
        "1  Configure & run",
        "2  Analyze results",
        "3  Compare prompts",
    )
    columns = st.columns(3)
    for index, (column, label) in enumerate(zip(columns, labels), start=1):
        if index == active_step:
            column.markdown(f"**:red[{label}]**")
        else:
            column.markdown(f"<span style='opacity:0.62'>{label}</span>", unsafe_allow_html=True)
    st.divider()


def openrouter_key_is_configured() -> bool:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return bool(key) and key.lower() not in _PLACEHOLDER_API_KEYS


def switch_page_button(label: str, target: str, *, primary: bool = False, key: str | None = None) -> None:
    if st.button(label, type="primary" if primary else "secondary", key=key):
        st.switch_page(target)
