"""Setup configuration for RAG Agent package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rag-agent",
    version="0.1.0",
    author="Marcos Oliveira",
    author_email="marcosf63@gmail.com",
    description="A strict document-based RAG agent using ChromaDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcosf63/rag-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "openai": ["openai"],
        "local": ["sentence-transformers"],
        "pdf": ["pypdf"],
    },
    entry_points={
        "console_scripts": [
            "rag-agent=rag_agent.examples.basic_usage:main",
        ],
    },
)