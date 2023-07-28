# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from datetime import datetime

# Project settings
project = "pika"
project_copyright = f"2022-{datetime.now().year}, ETH Zurich"
author = "ETH Zurich"
version = ""
release = version

# General sphinx settings
extensions = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst"]
language = "English"
primary_domain = "cpp"
highlight_language = "cpp"

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
    "base_url": "https://pika-org.github.io/pika-docs",
    "repo_url": "https://github.com/pika-org/pika",
    "repo_name": "pika",
    "html_minify": False,
    "html_prettify": True,
    "logo_icon": "&#xe88a;",
    "css_minify": True,
    "repo_type": "github",
    "globaltoc_depth": 2,
    "master_doc": False,
    "nav_links": [
        {"href": "index", "internal": True, "title": "Overview"},
        {"href": "usage", "internal": True, "title": "Usage"},
    ],
    "heroes": {
        "index": "Concurrency and parallelism built on C++ std::execution",
    },
    "version_dropdown": False,
    "table_classes": ["plain"],
}
