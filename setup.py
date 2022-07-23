from setuptools import find_packages, setup

setup(
    name="paddle",
    url="https://github.com/maxfrei750/paddle/",
    packages=find_packages(include=["paddle", "paddle.*"]),
    include_package_data=True,
)
