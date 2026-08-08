from __future__ import annotations

import hashlib
import hmac
import os

import streamlit as st


def check_password() -> bool:
    """Use the same optional application password on every Streamlit page."""
    expected = os.getenv("APP_PASSWORD", "").strip()
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("LLM Evaluation Platform")
    st.caption("Password-protected evaluation workspace")
    password = st.text_input("Application password", type="password")
    if st.button("Sign in", type="primary"):
        actual_digest = hashlib.sha256(password.encode()).digest()
        expected_digest = hashlib.sha256(expected.encode()).digest()
        if hmac.compare_digest(actual_digest, expected_digest):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False
