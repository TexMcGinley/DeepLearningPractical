from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dl_pipeline",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-alexnet=classification.alexnet:train",
            "train-resnet50=classification.resnet50:train",
        ],
    },
)