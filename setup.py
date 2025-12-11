from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="txt2sql",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Text-to-SQL using Small Language Models with LoRA, DoRA, and GRPO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TXT2SQL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "datasets>=2.15.0",
        "pandas>=2.1.4",
        "numpy>=1.24.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "black>=23.12.0",
            "flake8>=7.0.0",
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ],
    },
)
