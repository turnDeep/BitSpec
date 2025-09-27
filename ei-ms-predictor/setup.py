from setuptools import setup, find_packages

setup(
    name="ei-ms-predictor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="A machine learning model to predict EI mass spectra from chemical structures.",
    long_description=open("README.md").read(),
    author="Jules, AI Software Engineer",
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
