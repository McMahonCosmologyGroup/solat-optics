import os
import pathlib
from distutils.core import setup

import pkg_resources
import setuptools

VERSION = "0.1"


def build_packages(base_dir, name_base):
    """
    recusively find all the folders and treat them as packages
    """
    arr = [name_base]
    for fname in os.listdir(base_dir):
        if os.path.isdir(base_dir + fname):
            """
            ignore the hidden files
            """
            if fname[0] == ".":
                continue
            name = "{}.{}".format(name_base, fname)
            recursion = build_packages(base_dir + fname + "/", name)
            if len(recursion) != 0:
                [arr.append(rec) for rec in recursion]
    return arr


package = build_packages("solat_optics")

setup(
    name="solat_optics",
    version=VERSION,
    description="Optical modelling of the Simons Observatory Large Aperture Telescope.",
    author="Grace E. Chesmore",
    author_email="chesmore@uchicago.edu",
    package_dir={
        "holog_run": "holog_run",
        "ot_geo": "ot_geo",
        "ray_trace": "ray_trace",
        "ray_trace_int": "ray_trace_int",
    },
    packages=[package],
    scripts=["scripts/beamsim_set.sh", "scripts/latrt_holog_sim.ipynb"],
)


with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setuptools.setup(
    install_requires=install_requires,
)
