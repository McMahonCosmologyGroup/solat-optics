from setuptools import setup

setup(
    name="solat_optics",
    version="0.1.1",
    description="Optical modelling of the Simons Observatory Large Aperture Telescope.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Grace E. Chesmore, Alex Thomas, UChicago Lab",
    author_email="chesmore@uchicago.edu, agthomas@uchicago.edu",
    packages=["solat_optics"],
    package_dir={"solat_optics": "solat_optics"},
    install_requires=open("requirements.txt").read().splitlines()
)