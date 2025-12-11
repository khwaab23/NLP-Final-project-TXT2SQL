# Contributing to Text-to-SQL with Small Language Models

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## 🚀 How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU/CPU, etc.)
- **Error messages** or stack traces
- **Configuration files** if relevant

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear use case** for the feature
- **Detailed description** of the proposed functionality
- **Potential implementation approach** (if you have ideas)
- **Examples** of how it would be used

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Update documentation** as needed
5. **Submit a pull request**

## 🔧 Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/TXT2SQL.git
cd TXT2SQL
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Install Development Dependencies

```bash
pip install black flake8 mypy pytest pytest-cov
```

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 127 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Grouped (standard library, third-party, local)

### Code Formatting

Format your code with Black:

```bash
black src/ experiments/ tests/
```

### Linting

Check your code with flake8:

```bash
flake8 src/ experiments/ --max-line-length=127
```

### Type Hints

Add type hints to function signatures:

```python
def generate_sql(self, question: str, schema: str, db_id: str) -> str:
    """Generate SQL query from natural language question."""
    pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings explaining what you're testing

Example:

```python
def test_lora_model_initialization():
    """Test that LoRA model initializes correctly with default config."""
    model = PhiModel()
    assert model.model_type == "phi-2"
    assert model.model is not None
```

## 📚 Documentation

### Docstrings

Use Google-style docstrings:

```python
def train(self, num_epochs: int = 3, batch_size: int = 4) -> None:
    """Train the model with LoRA.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        None
        
    Raises:
        ValueError: If batch_size is less than 1
    """
    pass
```

### README Updates

If your changes affect usage, update:
- `README.md` - Main documentation
- `QUICKSTART.md` - Quick start guide
- `CHANGELOG.md` - Add entry for your changes

## 🎯 Areas for Contribution

### High Priority

- [ ] Additional SLM support (Gemma, Qwen, etc.)
- [ ] QLoRA implementation for 4-bit training
- [ ] More comprehensive test coverage
- [ ] Performance optimizations
- [ ] Bug fixes

### Medium Priority

- [ ] Additional datasets (Bird-SQL, KaggleDBQA)
- [ ] Multi-turn conversation support
- [ ] Web demo/UI
- [ ] Better error handling
- [ ] Documentation improvements

### Good First Issues

Look for issues labeled `good-first-issue` for beginner-friendly tasks:

- Documentation improvements
- Adding examples
- Simple bug fixes
- Code cleanup

## 🔄 Pull Request Process

### Before Submitting

1. ✅ **Code formatted** with Black
2. ✅ **Tests pass** locally
3. ✅ **Documentation updated**
4. ✅ **Type hints added**
5. ✅ **Commit messages** are clear

### PR Checklist

- [ ] PR has a descriptive title
- [ ] Changes are described in detail
- [ ] Related issues are linked
- [ ] Tests have been added/updated
- [ ] Documentation has been updated
- [ ] Code follows style guidelines
- [ ] All CI checks pass

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Approval** and merge

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add DoRA support for Mistral-7B model"
git commit -m "Fix memory leak in batch evaluation"
git commit -m "Update README with new installation instructions"

# Bad
git commit -m "fix bug"
git commit -m "update"
git commit -m "changes"
```

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## 🌳 Branching Strategy

- `main` - Stable release branch
- `develop` - Development branch
- `feature/xxx` - New features
- `bugfix/xxx` - Bug fixes
- `hotfix/xxx` - Urgent fixes

## 📦 Adding New Models

### Steps to Add a New SLM

1. Create model class in `src/models/slm_models.py`:

```python
class NewModel(SLMModel):
    def __init__(self):
        super().__init__(
            model_name="organization/model-name",
            model_type="new_model"
        )
```

2. Add configuration in `config/model_config.yaml`

3. Update documentation

4. Add tests in `tests/test_models.py`

5. Update README with model details

## 🐛 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing

**Import Errors:**
- Ensure virtual environment is activated
- Run `pip install -e .`

**Test Failures:**
- Check Python version compatibility
- Verify all dependencies installed
- Review test output carefully

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an issue with bug report template
- **Features**: Create an issue with feature request template
- **Chat**: Join our community (if applicable)

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in relevant documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Text-to-SQL with Small Language Models! 🎉**

Your contributions help make this project better for everyone.
