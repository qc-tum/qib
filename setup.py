from setuptools import setup, find_packages

setup(
    name="qib",
    version="0.1.0",
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "requests"
    ],
)
