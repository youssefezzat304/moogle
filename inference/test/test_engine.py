from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml

from lunar_clip.retrieval.indexing import (
    EmbeddingDescriptor,
    IndexArtifact,
    IndexManifest,
)
from lunar_clip.retrieval.indexing.artifacts import write_index_files
from lunar_data.catalog import (
    CaptionSelection,
    CatalogArtifact,
    CatalogManifest,
    CatalogRow,
    RasterGrid,
    SimpleCylindricalTransform,
)
from lunar_data.catalog.metadata import catalog_rows_to_frame
from moogle_inference import RetrievalEngine, load_retrieval_engine
from moogle_inference import loading as inference_loading


@dataclass
class FakeTextEmbedder:
    descriptor: EmbeddingDescriptor
    vector: torch.Tensor

    def __post_init__(self) -> None:
        self.queries: list[str] = []

    def encode(self, query: str) -> torch.Tensor:
        self.queries.append(query)
        return self.vector


@pytest.fixture
def descriptor() -> EmbeddingDescriptor:
    return EmbeddingDescriptor(
        model_id="fixture-model",
        checkpoint_sha256="a" * 64,
        modality="geomap",
        preprocessing_id="fixture-preprocessing",
        embedding_dimension=2,
    )


@pytest.fixture
def artifacts(
    tmp_path: Path,
    descriptor: EmbeddingDescriptor,
) -> tuple[CatalogArtifact, IndexArtifact]:
    catalog_root = tmp_path / "catalog"
    image_root = catalog_root / "images/wac/000"
    image_root.mkdir(parents=True)

    grid = RasterGrid(
        width=3,
        height=1,
        patch_size=1,
        stride=1,
    )
    transform = SimpleCylindricalTransform(
        origin_x_meters=-1.5,
        origin_y_meters=0.5,
        pixel_width_meters=1.0,
        pixel_height_meters=-1.0,
        radius_meters=100.0,
    )
    rows: list[CatalogRow] = []
    for patch_id in range(3):
        x_coord, y_coord = grid.origin_for_patch(patch_id)
        latitude, longitude = transform.patch_center(
            x_coord=x_coord,
            y_coord=y_coord,
            patch_size=grid.patch_size,
        )
        image_path = Path(f"images/wac/000/{patch_id}.webp")
        (catalog_root / image_path).write_bytes(b"fixture")
        rows.append(
            CatalogRow(
                patch_id=patch_id,
                x_coord=x_coord,
                y_coord=y_coord,
                latitude=latitude,
                longitude=longitude,
                description=f"Description {patch_id}",
                source_version="v3.0",
                prompt_style="Geologist to Non-Geologist",
                wac_image_path=str(image_path),
            )
        )

    catalog_manifest = CatalogManifest(
        catalog_id="tiny-lunar-v1",
        index_size=grid.patch_count,
        grid=grid,
        transform=transform,
        caption=CaptionSelection(),
        geomap_source="tiny-geomap",
        wac_source="tiny-wac",
    )
    catalog = CatalogArtifact(
        root=catalog_root,
        manifest=catalog_manifest,
        metadata=catalog_rows_to_frame(rows),
    )
    index = IndexArtifact(
        root=tmp_path / "index",
        manifest=IndexManifest(
            index_id="tiny-index-v1",
            catalog_id=catalog_manifest.catalog_id,
            index_size=grid.patch_count,
            descriptor=descriptor,
        ),
        patch_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    return catalog, index


def test_search_ranks_and_joins_catalog_metadata(
    artifacts: tuple[CatalogArtifact, IndexArtifact],
    descriptor: EmbeddingDescriptor,
) -> None:
    catalog, index = artifacts
    embedder = FakeTextEmbedder(
        descriptor=descriptor,
        vector=torch.tensor([[8.0, 6.0]]),
    )
    engine = RetrievalEngine(
        catalog=catalog,
        index=index,
        text_embedder=embedder,
    )

    results = engine.search("  young crater  ", top_k=3)

    assert embedder.queries == ["young crater"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.patch_id for result in results] == [0, 1, 2]
    assert [result.similarity for result in results] == pytest.approx([0.8, 0.6, -0.8])
    assert results[0].description == "Description 0"
    assert results[0].source_version == "v3.0"
    assert results[0].prompt_style == "Geologist to Non-Geologist"
    assert (
        results[0].wac_image_path == (catalog.root / "images/wac/000/0.webp").resolve()
    )


def test_empty_query_is_rejected_without_inference(
    artifacts: tuple[CatalogArtifact, IndexArtifact],
    descriptor: EmbeddingDescriptor,
) -> None:
    catalog, index = artifacts
    embedder = FakeTextEmbedder(
        descriptor=descriptor,
        vector=torch.tensor([1.0, 0.0]),
    )
    engine = RetrievalEngine(
        catalog=catalog,
        index=index,
        text_embedder=embedder,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        engine.search("   ")

    assert embedder.queries == []


@pytest.mark.parametrize("top_k", [0, 11, True, 1.5])
def test_invalid_top_k_is_rejected_before_inference(
    artifacts: tuple[CatalogArtifact, IndexArtifact],
    descriptor: EmbeddingDescriptor,
    top_k: object,
) -> None:
    catalog, index = artifacts
    embedder = FakeTextEmbedder(
        descriptor=descriptor,
        vector=torch.tensor([1.0, 0.0]),
    )
    engine = RetrievalEngine(
        catalog=catalog,
        index=index,
        text_embedder=embedder,
    )

    with pytest.raises((TypeError, ValueError)):
        engine.search("terrain", top_k=top_k)  # type: ignore[arg-type]

    assert embedder.queries == []


def test_engine_rejects_model_index_mismatch(
    artifacts: tuple[CatalogArtifact, IndexArtifact],
) -> None:
    catalog, index = artifacts
    incompatible = EmbeddingDescriptor(
        model_id="different-model",
        checkpoint_sha256="b" * 64,
        modality="geomap",
        preprocessing_id="fixture-preprocessing",
        embedding_dimension=2,
    )

    with pytest.raises(ValueError, match="does not match"):
        RetrievalEngine(
            catalog=catalog,
            index=index,
            text_embedder=FakeTextEmbedder(
                descriptor=incompatible,
                vector=torch.tensor([1.0, 0.0]),
            ),
        )


def test_production_loader_validates_and_composes_artifacts(
    tmp_path: Path,
    artifacts: tuple[CatalogArtifact, IndexArtifact],
    descriptor: EmbeddingDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, index = artifacts
    catalog.root.mkdir(exist_ok=True)
    (catalog.root / "manifest.yaml").write_text(
        yaml.safe_dump(catalog.manifest.to_dict(), sort_keys=False)
    )
    catalog.metadata.write_parquet(catalog.root / "metadata.parquet")
    write_index_files(
        index.root,
        manifest=index.manifest,
        patch_ids=index.patch_ids,
        embeddings=index.embeddings,
    )
    embedder = FakeTextEmbedder(
        descriptor=descriptor,
        vector=torch.tensor([1.0, 0.0]),
    )
    monkeypatch.setattr(
        inference_loading,
        "load_promoted_text_embedder",
        lambda manifest_path, *, device: embedder,
    )

    engine = load_retrieval_engine(
        catalog_path=catalog.root,
        index_path=index.root,
        model_manifest_path=tmp_path / "model/manifest.yaml",
        device="cpu",
    )

    assert engine.search("terrain", top_k=1)[0].patch_id == 0
