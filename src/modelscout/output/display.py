"""Compatibility shim: per-surface output modules now live alongside this file.

This module re-exports the public ``display_*`` functions so existing imports
(``from modelscout.output.display import display_ranking``) keep working. New
code should import from the specific submodule:

- ``modelscout.output.ranking`` for ranking + hardware tables
- ``modelscout.output.plan`` for the plan command
- ``modelscout.output.upgrade`` for the upgrade comparison
- ``modelscout.output.json_output`` for machine-readable JSON output
- ``modelscout.output.formatting`` for shared byte/param/date/color helpers
- ``modelscout.output._console`` for the shared Rich ``Console`` instance

The shared ``console`` symbol is re-exported here for read access. Code that
needs to *replace* the console (e.g. test capture) should set
``modelscout.output._console.console`` so every surface picks up the change.
"""

from modelscout.output._console import console
from modelscout.output.json_output import (
    display_json,
    display_plan_json,
    display_upgrade_json,
)
from modelscout.output.markdown import display_markdown
from modelscout.output.plan import display_plan
from modelscout.output.ranking import display_hardware, display_ranking
from modelscout.output.upgrade import display_upgrade

__all__ = [
    "console",
    "display_hardware",
    "display_json",
    "display_markdown",
    "display_plan",
    "display_plan_json",
    "display_ranking",
    "display_upgrade",
    "display_upgrade_json",
]
