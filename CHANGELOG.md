# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- QLoRA implementation for 4-bit training
- Additional SLM support (Gemma, Qwen)
- Multi-turn conversation support
- Web demo interface
- Additional datasets (Bird-SQL, KaggleDBQA)

## [0.1.0] - 2025-10-15

### Added
- Initial release of Text-to-SQL with Small Language Models
- Support for three SLMs: Phi-2 (2.7B), Llama-2-7B, Mistral-7B
- Three fine-tuning strategies:
  - LoRA (Low-Rank Adaptation)
  - DoRA (Weight-Decomposed Low-Rank Adaptation)
  - GRPO (Group Relative Policy Optimization)
- Integration with three generalist models:
  - GPT-4 Turbo (OpenAI API)
  - Gemini Pro (Google API)
  - Claude 3 Opus (Anthropic API)
- Comprehensive evaluation framework:
  - Exact Match metric
  - Execution Accuracy metric
  - Component Match metric
  - Token F1 Score metric
  - Valid SQL Rate metric
- Dataset support:
  - Spider dataset (10K+ examples, 200 databases)
  - WikiSQL dataset (80K+ examples)
- Data loading and preprocessing utilities
- Prompt engineering with multiple strategies:
  - Zero-shot prompting
  - Few-shot prompting
  - Chain-of-thought reasoning
  - Schema linking
- SQL execution and validation
- Training scripts for all strategies
- Automated evaluation and comparison scripts
- Interactive Jupyter notebooks:
  - Data exploration notebook
  - Results analysis notebook
- Shell scripts for automation:
  - Training automation (`scripts/train.sh`)
  - Evaluation automation (`scripts/evaluate.sh`)
  - Setup verification (`scripts/verify_setup.sh`)
- Comprehensive documentation:
  - README with full project overview
  - QUICKSTART guide for beginners
  - PROJECT_SUMMARY for quick reference
  - Configuration examples
- Development tools:
  - Black code formatting
  - Flake8 linting
  - Pytest testing framework
  - Type hints throughout codebase
- GitHub repository setup:
  - Issue templates (bug report, feature request)
  - Pull request template
  - CI/CD workflows (tests, code quality)
  - Contributing guidelines
  - MIT License

### Features
- 8-bit quantization for efficient training
- Gradient checkpointing for memory optimization
- Automatic checkpoint saving and resumption
- Batch evaluation for efficiency
- Cost tracking for API models
- Error analysis and categorization
- Performance comparison tables
- Reproducible experiments (fixed seeds)

### Documentation
- Complete API documentation with docstrings
- Usage examples for all components
- Configuration guides
- Troubleshooting section
- Best practices and tips

### Configuration
- Modular YAML-based configuration
- Separate configs for models and training
- Easy hyperparameter tuning
- API key management

---

## Version History

### Version Numbering

We use Semantic Versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Notes

**v0.1.0 - Initial Release**

This is the first public release of the Text-to-SQL with Small Language Models project. The goal is to demonstrate that small, specialized models can achieve competitive performance with large generalist models on the Text-to-SQL task.

**Key Highlights:**
- ✅ Complete implementation with 32+ files
- ✅ Three advanced fine-tuning strategies
- ✅ Comprehensive evaluation framework
- ✅ Production-ready code with proper error handling
- ✅ Full documentation and examples
- ✅ Automated testing and CI/CD

**What's Next:**
- See [Unreleased] section above for planned features
- Check [Issues](https://github.com/SaniyaGapchup/TXT2SQL/issues) for ongoing work
- Join discussions for feature requests

---

## How to Upgrade

### From Source

```bash
git pull origin main
pip install --upgrade -r requirements.txt
pip install -e .
```

### Migration Guides

No migration needed for initial release.

---

## Contributors

Thank you to all contributors who helped with this release!

- Initial implementation and design
- Documentation and examples
- Testing and bug fixes

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

---

[Unreleased]: https://github.com/SaniyaGapchup/TXT2SQL/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SaniyaGapchup/TXT2SQL/releases/tag/v0.1.0
