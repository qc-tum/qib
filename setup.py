from setuptools import setup

setup(
    name="qib",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src", "qib": "src/qib"},
    install_requires=[
        "numpy",
        "scipy",
        "importlib-metadata; python_version == '3.8'",
    ],
)
