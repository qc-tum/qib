from setuptools import setup, find_packages

setup(
    name="qib",
    version="0.0.1",
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
        "pyscf",
        "importlib-metadata; python_version == '3.8'",
    ],
)
