from setuptools import setup, find_packages

setup(
    name="sdm",
    version="0.0.1",
    license="MIT",
    description="Spatial Generalized Autoregressive Score Model",
    packages=find_packages(include=['sdm', 'sdm.*']),
    package_dir={"sdm": "./sdm"},
    install_requires=open("requirements.txt").read().split("\n"),
    python_requires=">=3.9",
)