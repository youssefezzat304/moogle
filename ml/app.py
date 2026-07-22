import streamlit as st

from src.lunar_data.loaders import load_dataset, load_dataframe
from src.lunar_apps.dashboard.ui.sidebar import render_sidebar
from src.lunar_apps.dashboard.ui.patch_view import render_patch_view
from src.lunar_apps.dashboard.ui.tokenizer_explorer import render_tokenizer_explorer
from src.lunar_apps.dashboard.ui.mlm_explorer import render_mlm_explorer

# Initialize page config
st.set_page_config(
    page_title='Lunar GeoData Explorer',
    layout='wide'
)

# 1. Load root dataset
with st.spinner("Loading lunar patch dataset..."):
    dataset = load_dataset()

# 2. Render sidebar and get user selections
sidebar_state = render_sidebar(len(dataset))

# 3. Stop if no valid results versions are available
if not sidebar_state.selected_version:
    st.stop()

# 4. Load the description dataframe
try:
    dataframe = load_dataframe(sidebar_state.selected_version)
except Exception as e:
    st.error(f"Could not load parquet for **{sidebar_state.selected_version}**: {e}")
    st.stop()

# 5. Render Patch View (returns the active caption for the tokenizer to analyze)
caption, selected_style = render_patch_view(
    dataset=dataset,
    patch_id=sidebar_state.patch_id,
    dataframe=dataframe,
    selected_version=sidebar_state.selected_version
)

# 6. Render Tokenizer Explorer
render_tokenizer_explorer(
    caption=caption,
    corpus_version=sidebar_state.selected_version,
    selected_style=selected_style
)

# 7. Render MLM Process Explorer
st.divider()
render_mlm_explorer(
    caption=caption,
    dataframe=dataframe,
    selected_style=selected_style,
    patch_id=sidebar_state.patch_id,
)
