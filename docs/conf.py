# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from datetime import datetime
import os

# Project settings
project = "pika"
project_copyright = f"2022-{datetime.now().year}, ETH Zurich"
author = "ETH Zurich"
version = ""
release = version

# General sphinx settings
extensions = ["breathe", "sphinx_immaterial", "recommonmark"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst"]
language = "English"
primary_domain = "cpp"
highlight_language = "cpp"
language = "en"

breathe_projects = {"pika": os.getenv("PIKA_DOCS_DOXYGEN_OUTPUT_DIRECTORY") + "/xml"}
breathe_default_project = "pika"
breathe_domain_by_extension = {
    "hpp": "cpp",
    "cpp": "cpp",
    "cu": "cpp",
}
breathe_default_members = ("members",)
breathe_show_include = False

show_warning_types = True

# HTML settings
html_title = "pika"
html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["pika.css"]
html_show_sourcelink = False
html_theme_options = {
    "site_url": "https://pikacpp.org",
    "repo_url": "https://github.com/pika-org/pika",
    "features": [
        "header.autohide",
        "navigation.instant",
        "navigation.tracking",
        "search.highlight",
        "search.share",
    ],
    "font": False,
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
}
