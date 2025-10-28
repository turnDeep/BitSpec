# setup.py
"""
マススペクトル予測パッケージのセットアップ
"""

from setuptools import setup, find_packages
import os

# バージョン情報
VERSION = "1.0.0"
DESCRIPTION = "AI-based Mass Spectrum Prediction for GC-MS"

# READMEの読み込み
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 依存関係の読み込み
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ms_predictor",
    version=VERSION,
    author="Your Name",
    author_email="your.email@example.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ms-predictor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ms-train=scripts.train:main",
            "ms-predict=scripts.predict:main",
            "ms-evaluate=scripts.evaluate:main",
        ],
    },
)
