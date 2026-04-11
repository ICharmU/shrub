from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import tempfile

import pandas as pd
import rasterio
from rasterio.windows import Window

from Final.pipeline_base import BasePipeline
from Final.models import (
    ModuleCard,
    PipelineDomain,
    RepresentationTarget,
    SpatialScope,
    ResolutionScope,
    AvailabilityTier,
    RuntimeTier,
    PipelineRunResult,
    CanonicalRasterOutputs,
    CanonicalObjectOutputs,
)
from Final.gating import (
    QACheckSpec,
    QACheckResult,
    ModuleQAProfile,
    ModuleQAEvaluation,
)
from Final.shared_utils import get_logger

from Final.labeling.sprint3_runner import (
    summarize_ptx_entries_by_site,
    select_ptx_entries,
    cleanup_stale_ptx_cache,
    download_ptx_with_cache,
    run_sprint3_for_ptx,
    append_results_manifest,
)
from Final.labeling.sprint3_standardize import standardize_sprint3_manifest
from Final.labeling.object_refinement import refine_shrub_objects
from Final.labeling.transforms import transform_objects_to_als, shrub_csv_to_transform_name
from Final.labeling.alignment import align_objects_to_naip
from Final.labeling.subspace_reduction import SubspaceReductionConfig
from Final.labeling.rasterize import (
    rasterize_objects,
    resample_single_band,
    write_single_band_geotiff,
)
from Final.labeling.dedup import deduplicate_artifact_table, extract_plot_key
from Final.labeling.qa import create_overlay_figure
from Final.labeling.export import export_table
from Final.labeling.io import download_file, extract_als_metadata
from Final.labeling.manifests import (
    list_files_with_suffix,
    site_to_remote_base,
    site_to_tif_name,
)


@dataclass
class LabelingPipelineConfig:
    sprint3_variant: str = "revised"
    use_shape_descriptors: bool = True
    use_temporal_confidence: bool = False
    use_boundary_confidence: bool = True
    use_transform_confidence: bool = False
    use_object_subspace_filter: bool = False
    rasterization_mode: str = "circle"
    multires: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    force_rerun_sprint4: bool = False
    force_refresh_site_assets: bool = False
    nonfatal_qa_overlay: bool = True

    # Sprint 3 execution / caching
    run_sprint3: bool = True
    sprint3_variants: tuple[str, ...] = ("original", "revised")
    max_ptx_per_site: int | None = 1
    force_rerun_sprint3: bool = False
    require_success_artifacts_sprint3: bool = True
    cleanup_ptx_after_all_variants: bool = True
    cleanup_stale_ptx_before_run: bool = True
    stale_ptx_days: int = 2

    boundary_confidence_mode: str = "radial"   # "radial" or "universal"
    site_reference_dates: dict[str, str] = field(default_factory=dict)

    subspace_min_component_pixels: int = 4
    subspace_min_object_confidence: float = 0.55
    subspace_min_transform_confidence: float = 0.50
    subspace_min_temporal_confidence: float = 0.40
    subspace_max_height_m: float = 3.5


@dataclass
class LabelingModuleSpec:
    key: str
    stage: str
    description: str
    enabled: bool = True
    submodules: list[str] = field(default_factory=list)


class LabelingPipeline(BasePipeline):
    def __init__(self, cfg, pipeline_config: LabelingPipelineConfig | None = None):
        super().__init__(
            cfg,
            pipeline_name="labeling",
            output_root=cfg.output.labeling_root / "pipeline_runs",
        )
        self.logger = get_logger("labeling.pipeline")
        self.pipeline_config = pipeline_config or LabelingPipelineConfig()

    # -------------------------------------------------------------------------
    # Paths / caches
    # -------------------------------------------------------------------------

    @property
    def summary_dir(self) -> Path:
        return self.cfg.output.labeling_root / "summaries"

    @property
    def sprint3_manifest_csv(self) -> Path:
        return self.cfg.output.labeling_root / "manifests" / self.cfg.labeling.sprint3_manifest_name

    @property
    def sprint4_manifest_csv(self) -> Path:
        return self.cfg.output.labeling_root / "manifests" / "sprint4_artifacts.csv"

    @property
    def site_cache_root(self) -> Path:
        root = self.cfg.output.labeling_root / "site_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_cache_root(self, site: str) -> Path:
        root = self.site_cache_root / site
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_transform_cache_root(self, site: str) -> Path:
        root = self._site_cache_root(site) / "transforms"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_transform_index_path(self, site: str) -> Path:
        return self._site_transform_cache_root(site) / "transform_index.json"

    def _site_naip_cache_root(self, site: str) -> Path:
        root = self._site_cache_root(site) / "naip"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _site_als_cache_root(self, site: str) -> Path:
        root = self._site_cache_root(site) / "als_metadata"
        root.mkdir(parents=True, exist_ok=True)
        return root
    
    @property
    def ptx_cache_root(self) -> Path:
        root = self.cfg.output.labeling_root / "ptx_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def sprint3_output_root(self) -> Path:
        return self.cfg.output.labeling_root

    # -------------------------------------------------------------------------
    # Stage 1: Sprint 3 manifest -> canonical objects
    # -------------------------------------------------------------------------

    def stage_run_sprint3(self) -> pd.DataFrame:
        """
        Discover PTX files, select the latest K per site, run configured Sprint 3
        variants with caching/cleanup, and update the Sprint 3 manifest CSV.
        """
        if not self.pipeline_config.run_sprint3:
            self.logger.info("Skipping Sprint 3 execution because run_sprint3=False")
            if self.sprint3_manifest_csv.exists():
                return pd.read_csv(self.sprint3_manifest_csv)
            return pd.DataFrame()

        self.sprint3_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        self.ptx_cache_root.mkdir(parents=True, exist_ok=True)

        if self.pipeline_config.cleanup_stale_ptx_before_run:
            stale_removed_df = cleanup_stale_ptx_cache(
                cache_root=self.ptx_cache_root,
                stale_days=self.pipeline_config.stale_ptx_days,
            )
            if not stale_removed_df.empty:
                self.logger.info("Removed %d stale PTX cache file(s)", len(stale_removed_df))

        ptx_summary_df = summarize_ptx_entries_by_site(
            cfg=self.cfg,
            site_ids=self.cfg.sites,
        )

        selected_ptx_df = select_ptx_entries(
            ptx_summary_df,
            max_ptx_per_site=self.pipeline_config.max_ptx_per_site,
        )

        self.logger.info(
            "Sprint 3 PTX discovery selected %d PTX row(s) across %d site(s)",
            len(selected_ptx_df),
            selected_ptx_df['site_id'].nunique() if not selected_ptx_df.empty else 0,
        )

        all_results = []

        for _, row in selected_ptx_df.iterrows():
            site = row["site_id"]
            ptx_entry = {
                "name": row["ptx_name"],
                "url": row["ptx_url"],
            }

            local_ptx = None
            all_variants_ok = True

            try:
                local_ptx = download_ptx_with_cache(
                    site_id=site,
                    ptx_entry=ptx_entry,
                    cache_root=self.ptx_cache_root,
                )
            except Exception:
                self.logger.exception("Failed to download PTX for site=%s entry=%s", site, ptx_entry)
                continue

            for variant in self.pipeline_config.sprint3_variants:
                self.logger.info(
                    "Starting Sprint 3 | site=%s | variant=%s | ptx=%s",
                    site, variant, local_ptx.name
                )
                try:
                    result = run_sprint3_for_ptx(
                        site_id=site,
                        ptx_path=local_ptx,
                        sprint3_base_dir=self.cfg.data.sprint3_base_dir,
                        output_root=self.sprint3_output_root,
                        variant=variant,
                        raise_on_error=True,
                        force_rerun=self.pipeline_config.force_rerun_sprint3,
                        require_success_artifacts=self.pipeline_config.require_success_artifacts_sprint3,
                    )
                    all_results.append(result)
                    append_results_manifest(self.sprint3_manifest_csv, [result])

                    self.logger.info(
                        "Completed Sprint 3 | site=%s | variant=%s | ptx=%s | cache=%s",
                        site,
                        variant,
                        local_ptx.name,
                        result.used_cache,
                    )
                except Exception:
                    all_variants_ok = False
                    self.logger.exception(
                        "Sprint 3 failed | site=%s | variant=%s | ptx=%s",
                        site, variant, local_ptx.name
                    )

            if (
                local_ptx is not None
                and self.pipeline_config.cleanup_ptx_after_all_variants
                and all_variants_ok
            ):
                try:
                    from Final.labeling.sprint3_runner import cleanup_ptx_file
                    cleanup_ptx_file(local_ptx)
                except Exception:
                    self.logger.exception("Failed PTX cleanup after Sprint 3 for %s", local_ptx)

        if self.sprint3_manifest_csv.exists():
            manifest_df = pd.read_csv(self.sprint3_manifest_csv)
        else:
            manifest_df = pd.DataFrame()

        self.logger.info(
            "Sprint 3 stage complete | manifest_rows=%d | manifest_csv=%s",
            len(manifest_df),
            self.sprint3_manifest_csv,
        )
        return manifest_df

    def stage_load_and_standardize_objects(
        self,
        manifest_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        manifest_csv = Path(manifest_csv) if manifest_csv is not None else self.sprint3_manifest_csv
        if not manifest_csv.exists():
            self.logger.warning("Sprint 3 manifest missing: %s", manifest_csv)
            return pd.DataFrame()

        runs_df = pd.read_csv(manifest_csv)
        if "returncode" in runs_df.columns:
            runs_df = runs_df[runs_df["returncode"] == 0].copy()

        # respect selected Sprint 3 variants from pipeline config
        if "variant" in runs_df.columns and self.pipeline_config.sprint3_variants:
            runs_df = runs_df[runs_df["variant"].isin(self.pipeline_config.sprint3_variants)].copy()

        objects = standardize_sprint3_manifest(
            runs_df,
            source_version_prefix="sprint3",
            label_variant="base",
            keep_only_valid_runs=True,
            require_success_returncode=True,
        )
        self.logger.info(
            "Standardized %d object rows from Sprint 3 manifest after variant filtering",
            len(objects),
        )
        return objects

    # -------------------------------------------------------------------------
    # Stage 2: refinement
    # -------------------------------------------------------------------------

    def stage_refine_objects(self, objects_df: pd.DataFrame) -> pd.DataFrame:
        if objects_df.empty:
            return objects_df.copy()

        subspace_cfg = SubspaceReductionConfig(
            min_component_pixels=self.pipeline_config.subspace_min_component_pixels,
            min_object_confidence=self.pipeline_config.subspace_min_object_confidence,
            min_transform_confidence=self.pipeline_config.subspace_min_transform_confidence,
            min_temporal_confidence=self.pipeline_config.subspace_min_temporal_confidence,
            max_height_m=self.pipeline_config.subspace_max_height_m,
        )

        refined = refine_shrub_objects(
            objects_df,
            self.cfg,
            site_reference_dates=self.pipeline_config.site_reference_dates,
            apply_subspace_filter=self.pipeline_config.use_object_subspace_filter,
            subspace_config=subspace_cfg,
        )
        self.logger.info("Refined %d object rows", len(refined))
        return refined

    # -------------------------------------------------------------------------
    # Stage 3: site asset prep
    # -------------------------------------------------------------------------

    def validate_cached_naip(self, naip_path: Path) -> bool:
        try:
            with rasterio.open(naip_path) as src:
                h = max(1, min(16, src.height))
                w = max(1, min(16, src.width))
                src.read([1], window=Window(0, 0, w, h))
            return True
        except Exception as e:
            self.logger.warning("Cached NAIP validation failed for %s: %s", naip_path, e)
            return False

    def prepare_site_assets(self, site: str, force_refresh: bool = False):
        site_base = site_to_remote_base(self.cfg, site)

        naip_name = site_to_tif_name(site)
        naip_local = self._site_naip_cache_root(site) / naip_name
        naip_url = f"{site_base}/{self.cfg.data.naip_3dep_dir}/{naip_name}"

        use_cached_naip = False
        if (not force_refresh) and naip_local.exists():
            if self.validate_cached_naip(naip_local):
                self.logger.info("Using cached NAIP for site=%s: %s", site, naip_local)
                use_cached_naip = True
            else:
                self.logger.warning("Deleting corrupt cached NAIP for site=%s: %s", site, naip_local)
                naip_local.unlink(missing_ok=True)

        if not use_cached_naip:
            self.logger.info("Downloading NAIP for site=%s to %s", site, naip_local)
            download_file(naip_url, naip_local)
            if not self.validate_cached_naip(naip_local):
                raise RuntimeError(f"Downloaded NAIP for site={site} is unreadable: {naip_local}")

        als_cache_dir = self._site_als_cache_root(site)
        als_meta_json = als_cache_dir / "als_metadata.json"

        if (not force_refresh) and als_meta_json.exists():
            self.logger.info("Using cached ALS metadata for site=%s: %s", site, als_meta_json)
            als_meta = json.loads(als_meta_json.read_text(encoding="utf-8"))
        else:
            als_url = f"{site_base}/{self.cfg.data.als_dir}"
            als_files = list_files_with_suffix(als_url, (".laz", ".las", ".copc.laz"))
            if not als_files:
                raise RuntimeError(f"No ALS files found for site '{site}' at {als_url}")

            self.logger.info("Downloading %d ALS file(s) for site=%s to extract metadata", len(als_files), site)

            records = []
            scratch_dir = Path(tempfile.mkdtemp(prefix=f"labeling_als_{site}_"))
            try:
                for entry in als_files:
                    local_path = scratch_dir / entry["name"]
                    download_file(entry["url"], local_path)
                    meta = extract_als_metadata(local_path)
                    meta["source_file"] = entry["name"]
                    records.append(meta)
                    try:
                        local_path.unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.warning("Failed to delete temporary ALS file %s: %s", local_path, e)
            finally:
                try:
                    scratch_dir.rmdir()
                except Exception:
                    pass

            als_meta_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
            als_meta = records
            self.logger.info("Cached ALS metadata for site=%s at %s", site, als_meta_json)

        return naip_local, als_meta

    # -------------------------------------------------------------------------
    # Stage 4: transform lookup / cache
    # -------------------------------------------------------------------------

    def get_site_transform_index(self, site: str, force_refresh: bool = False) -> dict:
        index_path = self._site_transform_index_path(site)

        if (not force_refresh) and index_path.exists():
            self.logger.info("Using cached transform index for site=%s: %s", site, index_path)
            return json.loads(index_path.read_text(encoding="utf-8"))

        remote_dir = f"{site_to_remote_base(self.cfg, site)}/{self.cfg.data.transformations_dir}"
        entries = list_files_with_suffix(remote_dir, (".txt",))
        index = {entry["name"]: entry["url"] for entry in entries}
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        self.logger.info("Cached transform index for site=%s with %d entries", site, len(index))
        return index

    def get_transform_local_cached(self, site: str, plot_id: str, force_refresh: bool = False) -> Path:
        transform_name = shrub_csv_to_transform_name(f"{plot_id}.csv")
        transform_local = self._site_transform_cache_root(site) / transform_name

        if (not force_refresh) and transform_local.exists():
            self.logger.info("Using cached transform for site=%s plot_id=%s", site, plot_id)
            return transform_local

        transform_index = self.get_site_transform_index(site, force_refresh=force_refresh)

        if transform_name in transform_index:
            transform_url = transform_index[transform_name]
            self.logger.info("Downloading exact-match transform for site=%s plot_id=%s", site, plot_id)
            download_file(transform_url, transform_local)
            return transform_local

        fallback_names = sorted(
            name for name in transform_index
            if name.startswith(plot_id) and name.endswith("toALS.txt")
        )
        if fallback_names:
            chosen = fallback_names[0]
            transform_url = transform_index[chosen]
            self.logger.warning(
                "Exact transform missing for site=%s plot_id=%s; using fallback transform %s",
                site, plot_id, chosen
            )
            download_file(transform_url, transform_local)
            return transform_local

        raise FileNotFoundError(
            f"No transform file found for site={site}, plot_id={plot_id}. "
            f"Expected exact name {transform_name} or fallback starting with {plot_id}."
        )

    # -------------------------------------------------------------------------
    # Stage 5: output paths / output cache
    # -------------------------------------------------------------------------

    def artifact_paths_for_plot(self, site: str, plot_id: str, source_version: str) -> dict:
        sv = str(source_version)

        masks_dir = self.cfg.output.labeling_root / "masks" / site / sv
        conf_dir = self.cfg.output.labeling_root / "confidence" / site / sv
        objid_dir = self.cfg.output.labeling_root / "object_id" / site / sv
        objects_dir = self.cfg.output.labeling_root / "objects" / site / sv
        qa_dir = self.cfg.output.labeling_root / "qa" / site / sv

        for d in [masks_dir, conf_dir, objid_dir, objects_dir, qa_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return {
            "binary_path": masks_dir / f"{plot_id}_mask.tif",
            "confidence_path": conf_dir / f"{plot_id}_confidence.tif",
            "object_id_path": objid_dir / f"{plot_id}_object_id.tif",
            "object_table_path": objects_dir / f"{plot_id}_objects.csv",
            "qa_path": qa_dir / f"{plot_id}_overlay.png",
            "masks_dir": masks_dir,
            "conf_dir": conf_dir,
        }

    def has_successful_sprint4_outputs(self, site: str, plot_id: str, source_version: str) -> bool:
        paths = self.artifact_paths_for_plot(site, plot_id, source_version)

        required = [
            paths["binary_path"],
            paths["confidence_path"],
            paths["object_id_path"],
            paths["object_table_path"],
            paths["qa_path"],
        ]
        if not all(p.exists() for p in required):
            return False

        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                continue
            multires_binary = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
            multires_conf = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"
            if not multires_binary.exists() or not multires_conf.exists():
                return False

        return True

    def build_artifact_rows_from_disk(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_df: pd.DataFrame,
    ) -> pd.DataFrame:
        paths = self.artifact_paths_for_plot(site, plot_id, source_version)

        date_token = objects_df["ptx_date_token"].iloc[0] if "ptx_date_token" in objects_df.columns and len(objects_df) else None
        variant = objects_df["variant"].iloc[0] if "variant" in objects_df.columns and len(objects_df) else None
        plot_key = extract_plot_key(plot_id)

        rows = []
        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                multires_binary_path = paths["binary_path"]
                multires_conf_path = paths["confidence_path"]
            else:
                multires_binary_path = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
                multires_conf_path = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"

            rows.append(
                {
                    "site_id": site,
                    "plot_id": plot_id,
                    "plot_key": plot_key,
                    "variant": variant,
                    "label_variant": "base",
                    "resolution_m": float(res),
                    "binary_mask_path": str(multires_binary_path),
                    "confidence_mask_path": str(multires_conf_path),
                    "object_id_raster_path": str(paths["object_id_path"]),
                    "object_table_path": str(paths["object_table_path"]),
                    "qa_overlay_path": str(paths["qa_path"]),
                    "n_objects": int(len(objects_df)),
                    "n_valid_objects": int(objects_df["valid_object"].sum()) if "valid_object" in objects_df.columns else int(len(objects_df)),
                    "date_token": date_token,
                    "source_version": source_version,
                }
            )

        return pd.DataFrame(rows)

    def append_sprint4_artifacts_manifest(self, new_artifacts_df: pd.DataFrame) -> pd.DataFrame:
        manifest_csv = self.sprint4_manifest_csv
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)

        if manifest_csv.exists():
            old_df = pd.read_csv(manifest_csv)
            combined = pd.concat([old_df, new_artifacts_df], ignore_index=True)
        else:
            combined = new_artifacts_df.copy()

        dedup_subset = ["site_id", "plot_id", "source_version", "resolution_m", "label_variant"]
        combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
        combined.to_csv(manifest_csv, index=False)
        return combined

    # -------------------------------------------------------------------------
    # Stage 6: one plot/source-version transfer
    # -------------------------------------------------------------------------

    def process_one_object_group_to_labels(
        self,
        *,
        site: str,
        plot_id: str,
        source_version: str,
        objects_group: pd.DataFrame,
        naip_path: Path | None,
        als_meta: list | None,
        force_rerun: bool = False,
    ):
        self.logger.info(
            "Processing Sprint 4 transfer | site=%s | plot_id=%s | source_version=%s | n_objects=%d",
            site, plot_id, source_version, len(objects_group)
        )

        if (not force_rerun) and self.has_successful_sprint4_outputs(site, plot_id, source_version):
            self.logger.info(
                "Using cached Sprint 4 outputs | site=%s | plot_id=%s | source_version=%s",
                site, plot_id, source_version
            )
            paths = self.artifact_paths_for_plot(site, plot_id, source_version)
            cached_objects = pd.read_csv(paths["object_table_path"])
            cached_artifacts = self.build_artifact_rows_from_disk(
                site=site,
                plot_id=plot_id,
                source_version=source_version,
                objects_df=cached_objects,
            )
            return cached_objects, cached_artifacts

        if naip_path is None or als_meta is None:
            raise ValueError("naip_path and als_meta are required when no Sprint 4 output cache is available.")

        transform_local = self.get_transform_local_cached(site, plot_id, force_refresh=force_rerun)

        objects = objects_group.copy()
        objects, tile = transform_objects_to_als(objects, transform_local, als_meta)

        tile_wkt = tile.get("srs_wkt")
        if not tile_wkt:
            raise ValueError(f"ALS tile {tile.get('source_file')} is missing CRS/WKT metadata.")

        objects, grid = align_objects_to_naip(objects, naip_path, tile_wkt, self.cfg)
        binary, confidence, object_id = rasterize_objects(
            objects,
            grid,
            self.cfg,
            boundary_mode=self.pipeline_config.boundary_confidence_mode,
            apply_mask_subspace_reduction=self.pipeline_config.use_object_subspace_filter,
            min_component_pixels=self.pipeline_config.subspace_min_component_pixels,
        )

        paths = self.artifact_paths_for_plot(site, plot_id, source_version)

        write_single_band_geotiff(paths["binary_path"], binary, grid, dtype="uint8", nodata=self.cfg.raster.background_value)
        write_single_band_geotiff(paths["confidence_path"], confidence, grid, dtype="float32", nodata=self.cfg.raster.confidence_background)
        write_single_band_geotiff(paths["object_id_path"], object_id, grid, dtype="int32", nodata=0)
        export_table(objects, paths["object_table_path"])

        try:
            create_overlay_figure(naip_path, paths["binary_path"], paths["qa_path"])
        except Exception as e:
            if self.pipeline_config.nonfatal_qa_overlay:
                self.logger.warning(
                    "QA overlay failed for site=%s plot_id=%s source_version=%s: %s",
                    site, plot_id, source_version, e
                )
            else:
                raise

        for res in self.pipeline_config.multires:
            if float(res) == 1.0:
                continue

            b_res, b_grid = resample_single_band(binary, grid, res)
            c_res, c_grid = resample_single_band(confidence, grid, res)

            multires_binary_path = paths["masks_dir"] / f"{plot_id}_mask_{res:g}m.tif"
            multires_conf_path = paths["conf_dir"] / f"{plot_id}_confidence_{res:g}m.tif"

            write_single_band_geotiff(multires_binary_path, b_res, b_grid, dtype="uint8", nodata=self.cfg.raster.background_value)
            write_single_band_geotiff(multires_conf_path, c_res, c_grid, dtype="float32", nodata=self.cfg.raster.confidence_background)

        artifacts_df = self.build_artifact_rows_from_disk(
            site=site,
            plot_id=plot_id,
            source_version=source_version,
            objects_df=objects,
        )

        self.logger.info(
            "Finished Sprint 4 transfer | site=%s | plot_id=%s | source_version=%s",
            site, plot_id, source_version
        )
        return objects, artifacts_df

    # -------------------------------------------------------------------------
    # Stage 7: full site loop
    # -------------------------------------------------------------------------

    def site_has_pending_sprint4_work(self, site: str, site_objects: pd.DataFrame) -> bool:
        grouped = site_objects.groupby(["plot_id", "source_version"], dropna=False)
        for (plot_id, source_version), _ in grouped:
            if not self.has_successful_sprint4_outputs(site, plot_id, source_version):
                return True
        return False

    def stage_transfer_all_sites(self, objects_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_site_objects = []
        all_site_artifacts = []

        for site in self.cfg.sites:
            self.logger.info("=" * 100)
            self.logger.info("Processing full Sprint 4 site loop for site=%s", site)

            site_objects = objects_df[objects_df["site_id"] == site].copy()
            if site_objects.empty:
                self.logger.info("No objects available for site=%s; skipping.", site)
                continue

            if (not self.pipeline_config.force_rerun_sprint4) and (not self.site_has_pending_sprint4_work(site, site_objects)):
                self.logger.info("All Sprint 4 outputs already cached for site=%s; skipping site asset prep.", site)

                grouped = site_objects.groupby(["site_id", "plot_id", "source_version"], dropna=False)
                for (_, plot_id, source_version), group_df in grouped:
                    cached_objects, cached_artifacts = self.process_one_object_group_to_labels(
                        site=site,
                        plot_id=plot_id,
                        source_version=source_version,
                        objects_group=group_df,
                        naip_path=None,
                        als_meta=None,
                        force_rerun=False,
                    )
                    all_site_objects.append(cached_objects)
                    all_site_artifacts.append(cached_artifacts)
                    self.append_sprint4_artifacts_manifest(cached_artifacts)
                continue

            try:
                naip_local, als_meta = self.prepare_site_assets(
                    site,
                    force_refresh=self.pipeline_config.force_refresh_site_assets,
                )

                grouped = site_objects.groupby(["site_id", "plot_id", "source_version"], dropna=False)
                for (_, plot_id, source_version), group_df in grouped:
                    try:
                        objects_out, artifacts_out = self.process_one_object_group_to_labels(
                            site=site,
                            plot_id=plot_id,
                            source_version=source_version,
                            objects_group=group_df,
                            naip_path=naip_local,
                            als_meta=als_meta,
                            force_rerun=self.pipeline_config.force_rerun_sprint4,
                        )
                        all_site_objects.append(objects_out)
                        all_site_artifacts.append(artifacts_out)
                        self.append_sprint4_artifacts_manifest(artifacts_out)
                    except Exception:
                        self.logger.exception(
                            "Sprint 4 failed | site=%s | plot_id=%s | source_version=%s",
                            site, plot_id, source_version
                        )
                        continue
            except Exception:
                self.logger.exception("Site-level Sprint 4 prep failed for site=%s", site)
                continue

        objects_all = pd.concat(all_site_objects, ignore_index=True) if all_site_objects else pd.DataFrame()
        artifacts_all = pd.concat(all_site_artifacts, ignore_index=True) if all_site_artifacts else pd.DataFrame()
        artifacts_all = deduplicate_artifact_table(artifacts_all)
        return objects_all, artifacts_all

    # -------------------------------------------------------------------------
    # Finalize / save
    # -------------------------------------------------------------------------

    def finalize_outputs(self, objects_df: pd.DataFrame, artifacts_df: pd.DataFrame) -> tuple[Path, Path]:
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        objects_csv = self.summary_dir / "objects_all.csv"
        artifacts_csv = self.summary_dir / "artifacts_all.csv"

        objects_df.to_csv(objects_csv, index=False)
        artifacts_df.to_csv(artifacts_csv, index=False)

        self.logger.info("Saved labeling summaries to %s and %s", objects_csv, artifacts_csv)
        return objects_csv, artifacts_csv

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(
        self,
        *,
        manifest_csv: str | Path | None = None,
        notes: list[str] | None = None,
    ) -> PipelineRunResult:
        notes = notes or []

        sprint3_manifest_df = self.stage_run_sprint3()
        manifest_source = manifest_csv if manifest_csv is not None else self.sprint3_manifest_csv

        objects_std = self.stage_load_and_standardize_objects(manifest_csv=manifest_source)
        objects_refined = self.stage_refine_objects(objects_std)
        objects_all, artifacts_all = self.stage_transfer_all_sites(objects_refined)
        objects_csv, artifacts_csv = self.finalize_outputs(objects_all, artifacts_all)

        module_specs = self.build_module_specs()
        qa_evals = self.evaluate_labeling_qc(objects_df=objects_all, artifacts_df=artifacts_all)

        success = (not objects_all.empty) and (not artifacts_all.empty)
        status = "success" if success else "empty_outputs"

        result = PipelineRunResult(
            pipeline_name=self.pipeline_name,
            success=success,
            status=status,
            raster_outputs=CanonicalRasterOutputs(
                labels=artifacts_all,
                qa_overlays=artifacts_all[["site_id", "plot_id", "qa_overlay_path"]].copy()
                if not artifacts_all.empty and "qa_overlay_path" in artifacts_all.columns
                else None,
            ),
            object_outputs=CanonicalObjectOutputs(
                objects=objects_all,
                source_provenance=objects_all[
                    [c for c in ["site_id", "plot_id", "source_version", "source_file"] if c in objects_all.columns]
                ].copy()
                if not objects_all.empty
                else None,
            ),
            qa_outputs={
                "objects_csv": str(objects_csv),
                "artifacts_csv": str(artifacts_csv),
                "artifacts_manifest_csv": str(self.sprint4_manifest_csv),
                "module_specs": {k: vars(v) for k, v in module_specs.items()},
                "module_qc": {
                    k: {
                        "pass_rate": v.pass_rate,
                        "mean_score": v.mean_score,
                        "results": [vars(r) for r in v.results],
                    }
                    for k, v in qa_evals.items()
                },
            },
            metrics={
                "n_object_rows": int(len(objects_all)),
                "n_artifact_rows": int(len(artifacts_all)),
                "n_sites": int(objects_all["site_id"].nunique()) if "site_id" in objects_all.columns and not objects_all.empty else 0,
            },
            notes=notes,
        )

        self.save_run_result(result)
        return result