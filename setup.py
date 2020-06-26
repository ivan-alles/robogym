#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import json
import os
import setuptools

with open(os.path.join(os.getcwd(), 'robogym', 'package.json'), 'r') as f:
    package = json.load(f)

install_requires = [
    "numpy"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=package['name'],
    version=package['version'],
    author=package['author'],
    description=package['description'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ivan-alles/" + package['name'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

