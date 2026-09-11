"""Include the catalogue when building from a checkout or source archive."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CatalogueBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        for asset_dir in (root / "assets", root.parent / "assets"):
            manifest = asset_dir / "dataset.json"
            if manifest.is_file():
                for asset in asset_dir.glob("*.json"):
                    build_data["force_include"][str(asset)] = f"modelscout/dataset/data/{asset.name}"
                return
        raise FileNotFoundError("The source catalogue assets are missing.")
