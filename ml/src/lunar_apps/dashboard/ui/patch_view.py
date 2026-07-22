import streamlit as st
import polars as pl

from lunar_apps.dashboard.utils import tensor_to_image, calculate_composition
from lunar_data.loaders import get_descriptions

def render_patch_view(dataset, patch_id: int, dataframe: pl.DataFrame, selected_version: str) -> tuple[str, str]:
    """Renders the patch data and returns the caption string and selected style."""
    sample           = dataset[patch_id]
    wac_tensor       = sample["wac"]["tensor"]
    geomap_tensor    = sample["geomap"]["original"]
    one_hot_tensor   = sample["geomap"]["tensor"]

    captions_dict    = get_descriptions(dataframe, patch_id)
    composition_data = calculate_composition(one_hot_tensor, dataset.legend)

    st.title(f"Viewing Patch #{patch_id}")
    st.caption(f"Description version: **{selected_version}**")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**WAC Image (Raw)**")
        st.image(tensor_to_image(wac_tensor), use_container_width=True)
    with img_col2:
        st.markdown("**Geomap (Labels)**")
        st.image(tensor_to_image(geomap_tensor), use_container_width=True)

    st.divider()

    st.subheader("Generated AI Caption")
    
    # --- Handle Multiple or Single Descriptions ---
    if "Error" in captions_dict:
        st.warning(captions_dict["Error"])
        selected_caption = captions_dict["Error"]
        selected_style = "Error"
    else:
        style_options = list(captions_dict.keys())
        
        if len(style_options) > 1:
            selected_style = st.radio(
                "Select Description Style:",
                options=style_options,
                horizontal=True
            )
        else:
            selected_style = style_options[0]

        selected_caption = captions_dict[selected_style]
        st.info(selected_caption)

    st.divider()

    st.subheader("Geological Composition")
    if composition_data:
        metric_cols = st.columns(len(composition_data))
        for i, data in enumerate(composition_data):
            with metric_cols[i]:
                st.metric(label=data['label'], value=f"{data['percentage']:.1f}%")
                st.markdown(
                    f'<div style="background:{data["color"]};height:8px;border-radius:4px;"></div>',
                    unsafe_allow_html=True
                )

        with st.expander("Full composition table", expanded=True):
            st.table(composition_data)
    else:
        st.info("No identifiable geological features in this patch.")

    st.divider()
    
    return selected_caption, selected_style