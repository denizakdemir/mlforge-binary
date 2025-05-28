from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="mlforge-binary",
    version="0.1.0",
    author="MLForge Team",
    author_email="team@mlforge.dev",
    description="A scikit-learn style library for binary classification that handles real-world challenges automatically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlforge/mlforge-binary",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cli": [
            "click>=8.0",
        ],
        "dashboard": [
            "streamlit>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlforge=mlforge_binary.cli:cli",
        ],
    },
)