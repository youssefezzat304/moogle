import streamlit as st
import polars as pl
from pathlib import Path
from tokenizers import Tokenizer

from lunar_data.lunar_geo_data import LunarGeoData
from lunar_apps.dashboard.config import CONFIG
from lunar_text.utils.tokenizers import load_tokenizer as load_tokenizer_file

@st.cache_resource
def load_dataset():
    return LunarGeoData(
        root=CONFIG['dataset']['path'],
        patch_size=CONFIG['hyperparameters']['patch_size'],
        stride=CONFIG['hyperparameters']['stride']
    )


def discover_versions(results_root: str = "results") -> list[str]:
    descriptions_path = Path(CONFIG["dataset"]["descriptions_path"])
    if descriptions_path.is_file():
        return [descriptions_path.stem]

    root = Path(results_root)
    if not root.exists():
        return []
    return sorted(
        [d.name for d in root.iterdir() if d.is_dir()],
        key=lambda v: [int(x) for x in v.lstrip("v").split(".") if x.isdigit()]
    )


def discover_tokenizer_versions(artifacts_root: str = "artifacts/tokenizers") -> list[str]:
    root = Path(artifacts_root)
    if not root.exists():
        return []
        
    valid_versions = []
    for alg_dir in root.iterdir():
        if alg_dir.is_dir():
            if (alg_dir / "tokenizer.json").exists() or (alg_dir / "tokenizer.csv").exists():
                valid_versions.append(alg_dir.name)
                
            for d in alg_dir.iterdir():
                if d.is_dir():
                    if (d / "tokenizer.json").exists() or (d / "tokenizer.csv").exists():
                        valid_versions.append(f"{alg_dir.name}/{d.name}")
                
    def sort_key(v):
        parts = v.split('/')
        alg = parts[0]
        if len(parts) > 1:
            nums = [int(x) for x in parts[1].lstrip("v").split(".") if x.isdigit()]
            return (alg, 1, nums)
        else:
            return (alg, 0, [])

    return sorted(valid_versions, key=sort_key)


@st.cache_data
def load_dataframe(version: str, results_root: str = "results") -> pl.DataFrame:
    descriptions_path = Path(CONFIG["dataset"]["descriptions_path"])
    if descriptions_path.is_file():
        return pl.read_parquet(descriptions_path)

    target_dir = Path(results_root) / version
    
    files = list(target_dir.rglob("*.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No .parquet files found in '{target_dir}' or its subdirectories.")
        
    return pl.scan_parquet(files).collect()


@st.cache_resource
def load_tokenizer(version: str) -> Tokenizer:
    return load_tokenizer_file(f"artifacts/tokenizers/{version}/tokenizer.json")


def get_corpus_texts(version: str, selected_style: str) -> list[str]:
    configured_path = Path(CONFIG["dataset"]["descriptions_path"])
    path_combined = Path(f"results/{version}/combined_captions.parquet")
    path_new = Path(f"results/{version}/generated_captions_merged.parquet")

    ps   = CONFIG['hyperparameters']['patch_size']
    s    = CONFIG['hyperparameters']['stride']
    path_old = Path(f"results/{version}/results_ps{ps}_s{s}.parquet")
    
    if configured_path.is_file():
        path = configured_path
    elif path_combined.exists():
        path = path_combined
    elif path_new.exists():
        path = path_new
    elif path_old.exists():
        path = path_old
    else:
        return []
        
    df = pl.read_parquet(path)

    if "text" in df.columns:
        text_df = df
        if "prompt_style" in df.columns:
            style_df = df.filter(pl.col("prompt_style") == selected_style)
            if not style_df.is_empty():
                text_df = style_df

        return (
            text_df.select(pl.col("text").cast(pl.Utf8).str.strip_chars().alias("text"))
            .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
            .get_column("text")
            .to_list()
        )
    
    target_col = f"caption_{selected_style}"
    if target_col in df.columns:
        return df.get_column(target_col).drop_nulls().to_list()

    if "llm_description" in df.columns:
        return df.get_column("llm_description").drop_nulls().to_list()
        
    return []


def get_descriptions(dataframe: pl.DataFrame, patch_id: int) -> dict[str, str]:
    # TODO: Standarize the patch_number or patch_id column in all datasets.
    if "patch_number" in dataframe.columns:
        result = dataframe.filter(pl.col("patch_number") == patch_id)
    elif "patch_id" in dataframe.columns:
        result = dataframe.filter(pl.col("patch_id") == patch_id)
    else:
        return {"Error": "Could not locate a patch ID column in this dataset."}
    
    if result.is_empty():
        return {"Error": "Text pipeline has not processed this patch yet."}

    descriptions = {}
    columns = result.columns

    if "text" in columns:
        if "prompt_style" in columns:
            for row in result.select(["prompt_style", "text"]).iter_rows(named=True):
                val = row.get("text")
                if val:
                    descriptions[str(row.get("prompt_style") or "Default")] = str(val)
        else:
            val = result.get_column("text").drop_nulls().head(1)
            if len(val) > 0 and val.item():
                descriptions["Default"] = str(val.item())

    if "llm_description" in columns:
        val = result.get_column("llm_description").item()
        if val:
            descriptions["Default"] = val

    for col in columns:
        if col.startswith("caption_"):
            style_name = col.replace("caption_", "")
            val = result.get_column(col).item()
            
            if val:
                descriptions[style_name] = val

    if not descriptions:
        return {"Error": "No descriptions found for this patch."}

    return descriptions
