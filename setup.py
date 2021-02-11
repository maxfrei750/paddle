#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To use: python setup.py install

import os
import re
from pathlib import Path
from typing import List

from setuptools import Command, setup
from setuptools.command.install import install

PACKAGE_NAME = "paddle"


def read_version(package_name: str) -> str:
    """Read version string from package main __init__.py file.

    Based on: https://packaging.python.org/guides/single-sourcing-package-version/

    :param package_name: name of the package
    :return: version string
    """

    setup_root = get_setup_root()

    with open(setup_root / package_name / "__init__.py") as file:
        version_file_content = file.read()

    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read_requirements() -> List[str]:
    """Read requirements from requirements.txt file.

    :return: list of requirements
    """
    setup_root = get_setup_root()

    with open(setup_root / "requirements.txt") as file:
        return file.read().splitlines()


def read_readme() -> str:
    """Read contents of the README.md file to use as long description.

    :return: content of the readme file.
    """
    setup_root = get_setup_root()

    with open(setup_root / "README.md", encoding="utf-8") as file:
        return file.read()


def read_license() -> str:
    """Read license name from first line of LICENSE file.

    :return: license name
    """
    setup_root = get_setup_root()

    with open(setup_root / "LICENSE") as file:
        first_license_line = file.readline()

    expected_line_end = " license \n"
    if not first_license_line.endswith(expected_line_end):
        RuntimeError(f"Expected first line of license file to end with: {expected_line_end}")

    return first_license_line[:-9]


def get_setup_root() -> Path:
    """Get absolute root path of the setup file.

    :return: absolute root path of the setup file
    """
    return Path(__file__).parent.resolve()


class CustomInstall(install):
    """"""

    def run(self):
        install.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    From https://stackoverflow.com/questions/3779915"""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def run():
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


setup(
    name=PACKAGE_NAME,
    version=read_version(PACKAGE_NAME),
    packages=[PACKAGE_NAME],
    install_requires=read_requirements(),
    url=f"https://github.com/maxfrei750/{PACKAGE_NAME}",
    license=read_license(),
    author="Max Frei",
    author_email="max.frei@uni-due.de",
    description="paddle (PArticle Detection via Deep LEarning)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    cmdclass={"clean": CleanCommand, "install": CustomInstall},
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
