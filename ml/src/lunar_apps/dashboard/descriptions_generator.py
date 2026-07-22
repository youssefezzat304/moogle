import json
import random
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from ollama import Client

from lunar_data.lunar_geo_data import LunarGeoData
from lunar_apps.dashboard.config import CONFIG
from lunar_apps.dashboard.prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE
from lunar_apps.dashboard.utils import (
    calculate_composition,
    format_composition_for_llm,
    _tensor_to_base64,
)
from lunar_apps.dashboard.time_tracker import TimeTracker
from lunar_apps.dashboard.storage import checkpoint, flush, last_completed_patch


MODEL_ID = "gemma4:e4b"
OLLAMA_HOST = "http://boba.local:11434"


def load_data(
    limit: int | None,
    start_padding: int,
    end_padding: int | None,
    seed: int,
) -> tuple[LunarGeoData, list[int]]:
    """
    Build the LunarGeoData dataset and return it together with
    the list of chosen patch indices.
    """
    random.seed(seed)

    dataset = LunarGeoData(
        root=CONFIG["dataset"]["path"],
        patch_size=CONFIG["hyperparameters"]["patch_size"],
        stride=CONFIG["hyperparameters"]["stride"],
    )

    data_length = len(dataset)
    end_idx = data_length if end_padding is None else data_length - end_padding
    idx_range = range(start_padding, end_idx)

    if limit is not None:
        chosen_indices = sorted(random.sample(idx_range, limit))
    else:
        chosen_indices = list(idx_range)

    return dataset, chosen_indices


def prepare_sample(
    sample: dict,
    dataset: LunarGeoData,
) -> list[dict]:
    """
    Build a single multimodal chat payload for Ollama from one dataset sample,
    using the locked-in production prompts.
    """
    comp_text = format_composition_for_llm(
        calculate_composition(sample["geomap"]["tensor"], dataset.legend)
    )

    user_text = USER_TEMPLATE.format(composition_text=comp_text)
    wac_b64 = _tensor_to_base64(sample["wac"]["tensor"])
    geomap_b64 = _tensor_to_base64(sample["geomap"]["original"])

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_text,
            "images": [wac_b64, geomap_b64],
        },
    ]


def run_inference(
    messages: list[dict],
    client: Client,
) -> str:
    """Send a single chat payload to Ollama and return the response text."""
    response = client.chat(
        model=MODEL_ID, 
        messages=messages,
        options={
            "temperature": CONFIG['generation']['temperature']
        }
    )
    text = response["message"]["content"]

    print(f"  ↳ {text[:80]}...")
    return text


def process_patches(
    dataset: LunarGeoData,
    chosen_indices: list[int],
    client: Client,
    tracker: TimeTracker,
    output_dir: str | Path,
    is_production_run: bool,
) -> list[dict]:
    """
    Run inference on all chosen patches using the active prompt.

    Production mode:
        checkpoint directly to Parquet every N samples

    Test mode:
        accumulate results in memory and return JSON-ready output
    """
    results = []
    buffer: list[dict] = []
    tracker_group = f"prompt_{PROMPT_VERSION}_samples"

    print(f"\n{'=' * 40}")
    print(f"Running Data Generation (Prompt: {PROMPT_VERSION})")
    print(f"{'=' * 40}")

    for patch_number in tqdm(
        chosen_indices,
        desc=f"Generation {PROMPT_VERSION}",
    ):
        with tracker.track("sample_preprocessing", group=tracker_group):
            sample = dataset[patch_number]
            messages = prepare_sample(sample, dataset)

        with tracker.track("sample_inference", group=tracker_group):
            text = run_inference(messages, client)

        with tracker.track("sample_saving", group=tracker_group):
            x = int(dataset.grid_points[patch_number, 0])
            y = int(dataset.grid_points[patch_number, 1])

            created_at = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            result = {
                "patch_number": patch_number,
                "x": x,
                "y": y,
                "patch_size": CONFIG["hyperparameters"]["patch_size"],
                "stride": CONFIG["hyperparameters"]["stride"],
                "prompt_version": PROMPT_VERSION,
                "llm_description": text.strip(),
                "created_at": created_at
            }

            if is_production_run:
                checkpoint(
                    buffer=buffer,
                    result=result,
                    every_n=50,
                    output_dir=output_dir,
                )
            else:
                results.append(result)

    if is_production_run:
        flush(buffer=buffer, output_dir=output_dir)

    return results


def run_generation_pipeline(
    limit: int | None,
    seed: int = 42,
    start_padding: int = 0,
    end_padding: int | None = None,
    output_dir: str | Path = "../results",
) -> None:
    tracker = TimeTracker(name="LunarGeo Production Pipeline")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_production_run = limit is None

    print(f"\n{'=' * 40}")
    if is_production_run:
        print("Initialising Storage & Resume State (Production Mode)...")
        completed_patch = last_completed_patch(output_dir)
    else:
        print(f"Random Sample Mode ({limit} patches). Storage disabled.")
        completed_patch = -1
    print(f"{'=' * 40}")

    print(f"\n{'=' * 40}")
    print("Loading the data...")
    print(f"{'=' * 40}")

    with tracker.track("data_loading"):
        dataset, chosen_indices = load_data(
            limit=limit,
            start_padding=start_padding,
            end_padding=end_padding,
            seed=seed,
        )

    if completed_patch >= 0:
        original_count = len(chosen_indices)
        chosen_indices = [
            idx for idx in chosen_indices if idx > completed_patch
        ]

        print(
            f" ↳ Resuming active run: "
            f"Skipped {original_count - len(chosen_indices)} completed patches."
        )

    if not chosen_indices:
        print(" ✓ All required patches already processed. Exiting.")
        return

    print(f"\n{'=' * 40}")
    print(f"Connecting to Ollama at {OLLAMA_HOST}")
    print(f"{'=' * 40}")

    with tracker.track("client_init"):
        client = Client(host=OLLAMA_HOST)

    results = process_patches(
        dataset=dataset,
        chosen_indices=chosen_indices,
        client=client,
        tracker=tracker,
        output_dir=output_dir,
        is_production_run=is_production_run,
    )

    print(f"\n{'=' * 40}")

    if not is_production_run:
        print("Writing random sample results to JSON...")

        output_filename = (
            f"results_for_{limit}_random_patches_{PROMPT_VERSION}.json"
        )
        output_file = output_dir / output_filename

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)

        print(f" ✓ Results written to {output_file}")

    print(f"{'=' * 40}")

    tracker.report()
    tracker.save_report(output_dir)
