#!/usr/bin/env python3
"""Refresh the checked-in planner view of the audited world catalog."""

from __future__ import annotations

import json

from or_audit.install.catalog import PLANNER_CATALOG_PATH, planner_catalog_data

PLANNER_CATALOG_PATH.write_text(
    json.dumps(planner_catalog_data(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
