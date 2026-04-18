from __future__ import annotations

from Final.models import ArtifactSpec, StorageTier


FE_ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "site_metadata_manifest": ArtifactSpec(
        key="site_metadata_manifest",
        rel_path_template="features/{site_id}/{config_signature}/site_assets/site_metadata_manifest.json",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "source_inventory": ArtifactSpec(
        key="source_inventory",
        rel_path_template="features/{site_id}/{config_signature}/site_assets/source_inventory.json",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "als_metadata_json": ArtifactSpec(
        key="als_metadata_json",
        rel_path_template="features/{site_id}/{config_signature}/site_assets/als_metadata.json",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "site_naip_raster": ArtifactSpec(
        key="site_naip_raster",
        rel_path_template="features/{site_id}/shared/site_assets/naip/{filename}",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=True,
    ),
    "site_3dep_raster": ArtifactSpec(
        key="site_3dep_raster",
        rel_path_template="features/{site_id}/shared/site_assets/3dep/{filename}",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=True,
    ),
    "site_rap_raster": ArtifactSpec(
        key="site_rap_raster",
        rel_path_template="features/{site_id}/shared/site_assets/rap/{filename}",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=True,
    ),
    "canonical_grid": ArtifactSpec(
        key="canonical_grid",
        rel_path_template="features/{site_id}/{config_signature}/canonical_grid/canonical_grid.json",
        storage_tier=StorageTier.LOCAL_ONLY,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "chunk_manifest": ArtifactSpec(
        key="chunk_manifest",
        rel_path_template="features/{site_id}/{config_signature}/chunk_manifest/chunk_manifest.json",
        storage_tier=StorageTier.LOCAL_ONLY,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "source_ready_manifest": ArtifactSpec(
        key="source_ready_manifest",
        rel_path_template="features/{site_id}/{config_signature}/{source_name}/source_ready.json",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "family_chunk_npz": ArtifactSpec(
        key="family_chunk_npz",
        rel_path_template="features/{site_id}/{config_signature}/{source_name}/{family_name}/{chunk_id}.npz",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=False,
        prune_local_after_push=True,
    ),
    "stack_registry": ArtifactSpec(
        key="stack_registry",
        rel_path_template="features/{site_id}/{config_signature}/stack/stack_registry.json",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
    "object_feature_table": ArtifactSpec(
        key="object_feature_table",
        rel_path_template="features/{site_id}/{config_signature}/objects/object_features.csv",
        storage_tier=StorageTier.LOCAL_THEN_REMOTE,
        required_for_resume=True,
        prune_local_after_push=False,
    ),
}