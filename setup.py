from distutils.core import setup
from pathlib import Path
from typing import List

from setuptools import find_packages


def find_requirements() -> List[str]:
    requirements_txt = Path(__file__).parent / "requirement.txt"
    requirements = []
    if requirements_txt.exists():
        with open(requirements_txt, "r", encoding="utf-8") as req_file:
            for line in req_file:
                requirements.append(line.strip())
    return requirements


setup(
    name="adas",
    version="0.1",
    description="ADAS",
    author="Sergei Porkhun",
    author_email="ser.porkhun41@gmail.com",
    packages=find_packages(),
    install_requires=find_requirements(),
)
