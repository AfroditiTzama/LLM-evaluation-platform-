from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from database import ensure_database_file, get_database_path
from llm_eval.streamlit_auth import check_password
from llm_eval.streamlit_pages import render_prompt_comparison
from llm_eval.streamlit_ui import apply_ui_style, render_sidebar_navigation


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=True)

st.set_page_config(page_title="Prompt Comparison · LLM Evaluation", layout="wide")
if not check_password():
    st.stop()

apply_ui_style()
render_sidebar_navigation("comparison")
DB_PATH = ensure_database_file(get_database_path())
render_prompt_comparison(db_path=DB_PATH)
