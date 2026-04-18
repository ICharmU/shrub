from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil
import time
import yaml

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except Exception:  # pragma: no cover
    GoogleAuth = None
    GoogleDrive = None


# -----------------------------------------------------------------------------
# Generic artifact-store interface
# -----------------------------------------------------------------------------


class ArtifactStore(ABC):
    @abstractmethod
    def exists(self, rel_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pull(self, rel_path: str, local_path: str | Path | None = None) -> Path:
        raise NotImplementedError

    @abstractmethod
    def push(self, local_path: str | Path, rel_path: str | None = None) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, rel_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def sync_registry(self) -> None:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Local filesystem store
# -----------------------------------------------------------------------------


@dataclass
class LocalArtifactStore(ArtifactStore):
    repo_root: Path
    storage_root: Path

    def _abs_remote(self, rel_path: str) -> Path:
        return self.storage_root / rel_path

    def exists(self, rel_path: str) -> bool:
        return self._abs_remote(rel_path).exists()

    def pull(self, rel_path: str, local_path: str | Path | None = None) -> Path:
        src = self._abs_remote(rel_path)
        if not src.exists():
            raise FileNotFoundError(f"LocalArtifactStore missing remote artifact: {src}")

        dst = Path(local_path) if local_path is not None else (self.repo_root / rel_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst

    def push(self, local_path: str | Path, rel_path: str | None = None) -> str | None:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Cannot push missing artifact: {local_path}")

        rel_path = rel_path or str(local_path.relative_to(self.repo_root))
        dst = self._abs_remote(rel_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != dst.resolve():
            shutil.copy2(local_path, dst)
        return str(dst)

    def delete(self, rel_path: str) -> None:
        p = self._abs_remote(rel_path)
        if p.exists() and p.is_file():
            p.unlink()

    def sync_registry(self) -> None:
        return


# -----------------------------------------------------------------------------
# Drive-backed registry store
# -----------------------------------------------------------------------------


@dataclass
class DriveRegistryArtifactStore(ArtifactStore):
    repo_root: Path
    registry_path: Path
    drive_config_path: Path
    client_secrets_path: Path
    credentials_path: Path
    artifact_root_name: str = "artifacts"

    _registry: dict[str, Any] | None = None
    _drive: Any = None
    _drive_root_id: str | None = None

    # -------------------------- registry -------------------------------------

    def _load_registry(self, force: bool = False) -> dict[str, Any]:
        if self._registry is not None and not force:
            return self._registry

        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._registry = yaml.safe_load(f) or {}
        else:
            self._registry = {
                "drive_root_folder_id": None,
                "files": {},
            }
        return self._registry

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._registry or {}, f, sort_keys=True)

    def sync_registry(self) -> None:
        # lightweight for now: just reload from disk
        self._load_registry(force=True)

    # -------------------------- drive auth -----------------------------------

    def _load_drive_root_id(self) -> str:
        with open(self.drive_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        root_id = cfg.get("drive_root_folder_id")
        if not root_id:
            raise ValueError("drive_root_folder_id missing in drive_config.yaml")
        return str(root_id)

    def _get_drive_and_root(self):
        if self._drive is not None and self._drive_root_id is not None:
            return self._drive, self._drive_root_id

        if GoogleAuth is None or GoogleDrive is None:
            raise ImportError("pydrive2 is required for DriveRegistryArtifactStore")

        gauth = GoogleAuth()
        
        gauth.LoadClientConfigFile(str(self.client_secrets_path))

        if self.credentials_path.exists():
            gauth.LoadCredentialsFile(str(self.credentials_path))

        gauth.settings['get_refresh_token'] = True

        if gauth.credentials is None:
            # No credentials exist yet
            print("[artifact_store] Performing Google OAuth (one-time).")
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Token exists but is expired. Refresh it quietly!
            print("[artifact_store] Refreshing expired Google Drive credentials.")
            gauth.Refresh()
        else:
            # Token exists and is valid
            print("[artifact_store] Using cached Google Drive credentials.")
            gauth.Authorize()
            
        # Always save to ensure updated tokens (like new access tokens) are cached
        gauth.SaveCredentialsFile(str(self.credentials_path))

        drive = GoogleDrive(gauth)
        root_id = self._load_drive_root_id()

        self._drive = drive
        self._drive_root_id = root_id
        return self._drive, self._drive_root_id

    # -------------------------- drive path ops --------------------------------

    @staticmethod
    def _q_escape(s: str) -> str:
        return s.replace("'", "\\'")

    def _resolve_rel_path(self, rel_path: str) -> tuple[list[str], str]:
        rel = Path(rel_path)
        parts = rel.parts
        if not parts:
            raise ValueError("rel_path cannot be empty")
        return list(parts[:-1]), parts[-1]

    def _safe_list_files(self, drive, query: str):
        """
        Robust wrapper around PyDrive2 ListFile/GetList().
        Handles older 'items' style payloads, newer 'files' style payloads,
        and error-shaped responses more gracefully.
        """
        try:
            file_list = drive.ListFile({"q": query})
            result = file_list.GetList()
            return result
        except KeyError as e:
            meta = getattr(file_list, "metadata", None)
            if isinstance(meta, dict):
                if "files" in meta and isinstance(meta["files"], list):
                    return meta["files"]
                if "error" in meta:
                    raise RuntimeError(f"Drive ListFile returned error payload: {meta['error']}")
                raise RuntimeError(f"Drive ListFile returned unexpected metadata keys: {list(meta.keys())}") from e
            raise
        except Exception:
            raise

    def _ensure_remote_folder_chain(self, drive, root_id: str, folder_parts: list[str]) -> str:
        parent_id = root_id
        for folder_name in folder_parts:
            folder_name_q = self._q_escape(folder_name)
            query = (
                f"'{parent_id}' in parents and "
                f"title = '{folder_name_q}' and "
                "mimeType = 'application/vnd.google-apps.folder' and trashed=false"
            )
            folder_list = self._safe_list_files(drive, query)
            if folder_list:
                parent_id = folder_list[0]["id"]
            else:
                folder = drive.CreateFile(
                    {
                        "title": folder_name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [{"id": parent_id}],
                    }
                )
                folder.Upload()
                parent_id = folder["id"]
        return parent_id

    def _lookup_remote_file(self, rel_path: str):
        drive, root_id = self._get_drive_and_root()
        folder_parts, filename = self._resolve_rel_path(rel_path)

        parent_id = root_id
        for folder_name in folder_parts:
            folder_name_q = self._q_escape(folder_name)
            query = (
                f"'{parent_id}' in parents and "
                f"title = '{folder_name_q}' and "
                "mimeType = 'application/vnd.google-apps.folder' and trashed=false"
            )
            flist = self._safe_list_files(drive, query)
            if not flist:
                return None, None
            parent_id = flist[0]["id"]

        filename_q = self._q_escape(filename)
        query = (
            f"'{parent_id}' in parents and "
            f"title = '{filename_q}' and trashed=false"
        )
        files = self._safe_list_files(drive, query)
        if not files:
            return drive, None
        return drive, files[0]

    # -------------------------- public API ------------------------------------

    def exists(self, rel_path: str) -> bool:
        reg = self._load_registry()
        files = reg.setdefault("files", {})
        if rel_path in files and files[rel_path].get("drive_file_id"):
            return True

        _, gfile = self._lookup_remote_file(rel_path)
        return gfile is not None

    def pull(self, rel_path: str, local_path: str | Path | None = None) -> Path:
        reg = self._load_registry()
        files = reg.setdefault("files", {})
    
        dst = Path(local_path) if local_path is not None else (self.repo_root / rel_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
    
        file_id = files.get(rel_path, {}).get("drive_file_id")
    
        drive, gfile = self._get_drive_and_root()[0], None
        if file_id:
            gfile = drive.CreateFile({"id": file_id})
        else:
            drive, gfile = self._lookup_remote_file(rel_path)
    
        if gfile is None:
            raise FileNotFoundError(f"No remote Drive artifact for {rel_path}")
    
        # Important: force a fresh download, do not try to resume into stale staging files
        # if dst.exists():
        #     try:
        #         dst.unlink()
        #     except Exception:
        #         pass
    
        # tmp_dst = dst.with_suffix(dst.suffix + ".download")
    
        # if tmp_dst.exists():
        #     try:
        #         tmp_dst.unlink()
        #     except Exception:
        #         pass
    
        # try:
        #     gfile.GetContentFile(str(tmp_dst))
        #     tmp_dst.replace(dst)
        # finally:
        #     if tmp_dst.exists():
        #         try:
        #             tmp_dst.unlink()
        #         except Exception:
        #             pass

        self._download_to_path(gfile, dst, max_attempts=3, sleep_sec=1.0)
    
        entry = files.setdefault(rel_path, {})
        entry["local_path"] = rel_path
        entry["drive_file_id"] = gfile["id"]
        self._save_registry()
        return dst

    def push(self, local_path: str | Path, rel_path: str | None = None) -> str | None:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Cannot push missing artifact: {local_path}")

        rel_path = rel_path or str(local_path.relative_to(self.repo_root))
        drive, root_id = self._get_drive_and_root()
        folder_parts, filename = self._resolve_rel_path(rel_path)
        parent_id = self._ensure_remote_folder_chain(drive, root_id, folder_parts)

        filename_q = self._q_escape(filename)
        query = (
            f"'{parent_id}' in parents and "
            f"title = '{filename_q}' and trashed=false"
        )
        existing = drive.ListFile({"q": query}).GetList()

        if existing:
            gfile = existing[0]
            print(f"[artifact_store] Updating existing Drive artifact for {rel_path}")
        else:
            gfile = drive.CreateFile({"title": filename, "parents": [{"id": parent_id}]})
            print(f"[artifact_store] Creating new Drive artifact for {rel_path}")

        gfile.SetContentFile(str(local_path))
        gfile.Upload()
        file_id = gfile["id"]

        reg = self._load_registry()
        files = reg.setdefault("files", {})
        entry = files.setdefault(rel_path, {})
        entry["local_path"] = rel_path
        entry["drive_file_id"] = file_id
        self._save_registry()

        return file_id

    def delete(self, rel_path: str) -> None:
        drive, gfile = self._lookup_remote_file(rel_path)
        if gfile is not None:
            gfile.Delete()

        reg = self._load_registry()
        files = reg.setdefault("files", {})
        if rel_path in files:
            del files[rel_path]
            self._save_registry()

    def _download_to_path(self, gfile, dst: Path, *, max_attempts: int = 3, sleep_sec: float = 1.0) -> Path:
        last_err = None
    
        for attempt in range(1, max_attempts + 1):
            tmp_dst = dst.with_suffix(dst.suffix + f".download.{attempt}")
    
            try:
                if dst.exists():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
    
                if tmp_dst.exists():
                    try:
                        tmp_dst.unlink()
                    except Exception:
                        pass
    
                gfile.GetContentFile(str(tmp_dst))
                tmp_dst.replace(dst)
                return dst
    
            except Exception as e:
                last_err = e
                try:
                    tmp_dst.unlink(missing_ok=True)
                except Exception:
                    pass
    
                if attempt < max_attempts:
                    time.sleep(sleep_sec)
    
        raise last_err

# -----------------------------------------------------------------------------
# Hybrid wrapper
# -----------------------------------------------------------------------------


@dataclass
class HybridArtifactStore(ArtifactStore):
    local_store: LocalArtifactStore
    remote_store: ArtifactStore | None = None

    def exists(self, rel_path: str) -> bool:
        return self.local_store.exists(rel_path) or (self.remote_store.exists(rel_path) if self.remote_store else False)

    def pull(self, rel_path: str, local_path: str | Path | None = None) -> Path:
        if self.local_store.exists(rel_path):
            return self.local_store.pull(rel_path, local_path=local_path)
    
        if self.remote_store is None:
            raise FileNotFoundError(f"No local artifact and no remote store for {rel_path}")
    
        pulled = self.remote_store.pull(rel_path, local_path=local_path)
    
        # try:
        #     self.local_store.push(pulled, rel_path=rel_path)
        # except Exception:
        #     pass
    
        return pulled

    def push(self, local_path: str | Path, rel_path: str | None = None) -> str | None:
        if self.remote_store is not None:
            try:
                return self.remote_store.push(local_path, rel_path=rel_path)
            except Exception:
                pass
        return self.local_store.push(local_path, rel_path=rel_path)

    def delete(self, rel_path: str) -> None:
        self.local_store.delete(rel_path)
        if self.remote_store:
            self.remote_store.delete(rel_path)

    def sync_registry(self) -> None:
        if self.remote_store:
            self.remote_store.sync_registry()