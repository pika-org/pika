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
extensions = ["breathe", "recommonmark"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst"]
language = "English"
primary_domain = "cpp"
highlight_language = "cpp"

breathe_projects = {"pika": os.getenv("PIKA_DOCS_DOXYGEN_OUTPUT_DIRECTORY") + "/xml"}
breathe_default_project = "pika"
breathe_domain_by_extension = {
    "hpp": "cpp",
    "cpp": "cpp",
    "cu": "cpp",
}
breathe_default_members = ("members",)
breathe_show_include = False

# HTML settings
html_title = "pika"
html_theme = "sphinx_material"
html_static_path = ["_static"]
html_css_files = ["pika.css"]
html_show_sourcelink = False
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}
html_theme_options = {
    "nav_title": "pika",
    "color_primary": "blue-grey",
    "color_accent": "orange",
    "base_url": "https://pikacpp.org",
    "repo_url": "https://github.com/pika-org/pika",
    "repo_name": "pika",
    "html_minify": False,
    # Disabled because of https://github.com/bashtage/sphinx-material/issues/123
    "html_prettify": False,
    "logo_icon": "&#xe88a;",
    "css_minify": True,
    "repo_type": "github",
    "globaltoc_depth": 2,
    "master_doc": False,
    "nav_links": [
        {"href": "index", "internal": True, "title": "Overview"},
        {"href": "usage", "internal": True, "title": "Usage"},
        {"href": "api", "internal": True, "title": "API reference"},
        {"href": "changelog", "internal": True, "title": "Changelog"},
    ],
    "heroes": {
        "index": "Concurrency and parallelism built on C++ std::execution",
    },
    "version_dropdown": False,
    "table_classes": ["plain"],
}
