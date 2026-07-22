import streamlit as st
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from lunar_apps.dashboard.config import CONFIG
from lunar_data.loaders import discover_versions

@dataclass
class SidebarState:
    patch_id: int
    selected_version: str | None

def randomize_patch(dataset_len: int):
    st.session_state.p_index = np.random.randint(0, dataset_len)

def render_sidebar(dataset_len: int) -> SidebarState:
    if 'p_index' not in st.session_state:
        st.session_state.p_index = 0

    with st.sidebar:
        st.info(f"""
            **Dataset Configuration:**
            * **Total Patches:** {dataset_len:,}
            * **Patch Size:** {CONFIG['hyperparameters']['patch_size']} px
            * **Stride:** {CONFIG['hyperparameters']['stride']} px
        """)

        st.header("Dataset Navigation")
        patch_id = st.number_input(
            "Patch Index", min_value=0, max_value=dataset_len, key="p_index"
        )
        st.button("Random Patch", on_click=randomize_patch, args=(dataset_len,))

        st.divider()

        # Description version
        st.header("LLM Description Version")
        versions = discover_versions()

        if versions:
            selected_version = st.selectbox(
                "Results dataset version",
                options=versions,
                index=len(versions) - 1,
                help="Versioned result sets are loaded from results/<version>/",
            )
            descriptions_path = Path(CONFIG["dataset"]["descriptions_path"])
            st.caption(f"Loaded from: `{descriptions_path}`")
        else:
            st.warning("No versioned result folders found under `results/`.")
            selected_version = None

    return SidebarState(
        patch_id=patch_id,
        selected_version=selected_version
    )
