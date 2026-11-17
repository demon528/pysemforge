"""Temporary setup.py for building the sdist/wheel in environments where
hatchling's isolated build fails (e.g. missing python venv).
The published package itself uses pyproject.toml + hatchling as declared in
pyproject.toml; this file is only a build shim.
"""
import os
import re

from setuptools import setup, find_packages

here = os.path.dirname(os.path.abspath(__file__))

# Pull metadata from pyproject.toml so we stay single-source.
with open(os.path.join(here, "pyproject.toml")) as f:
    toml = f.read()

def get(field):
    m = re.search(rf'^{field}\s*=\s*"([^"]+)"', toml, re.MULTILINE)
    return m.group(1) if m else None

def get_list(field):
    m = re.search(rf'^{field}\s*=\s*\[(.*?)\]', toml, re.MULTILINE | re.DOTALL)
    if not m:
        return []
    return [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]

deps = get_list("dependencies")
python_requires = get("requires-python")

# README
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_desc = f.read()

# Use the src/ layout: tell setuptools the package lives in src/pysemforge.
setup(
    name="pysemforge",
    version=get("version"),
    description=get("description"),
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="MIT",
    author=get("name"),
    python_requires=python_requires,
    install_requires=deps,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"pysemforge": ["py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
)
