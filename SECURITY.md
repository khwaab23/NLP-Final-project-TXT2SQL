# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Text-to-SQL seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do NOT:

- Open a public GitHub issue for security vulnerabilities
- Discuss the vulnerability in public forums, social media, or mailing lists

### Please DO:

1. **Email us directly** at: security@yourdomain.com (replace with your actual email)
2. **Include details**:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (optional)

### What to expect:

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Investigation**: We'll investigate and determine severity
- **Fix**: We'll work on a fix and coordinate disclosure
- **Credit**: We'll credit you for responsible disclosure (if desired)

## Security Best Practices

### API Keys

- **Never commit API keys** to the repository
- Use environment variables or `config/api_keys.yaml` (which is gitignored)
- Rotate API keys regularly
- Use read-only API keys when possible

### Dependencies

- Keep dependencies up to date
- Review `requirements.txt` regularly
- Use `pip-audit` to check for known vulnerabilities:
  ```bash
  pip install pip-audit
  pip-audit
  ```

### Data Security

- **Do not upload sensitive databases** to public repositories
- Sanitize SQL queries before logging
- Be cautious with database credentials
- Use separate databases for development and production

### Model Security

- Verify model checksums before use
- Use official HuggingFace repositories only
- Be aware of prompt injection attacks
- Sanitize user inputs

## Known Security Considerations

### 1. SQL Injection (Model-Generated)

While this project generates SQL from natural language, the generated SQL should **always be validated and sanitized** before execution on production databases.

**Mitigation**:
- Use the `SQLExecutor` class which includes validation
- Run generated queries in read-only mode
- Use database permissions appropriately
- Never execute on production databases without review

### 2. API Key Exposure

**Mitigation**:
- API keys are stored in `config/api_keys.yaml` (gitignored)
- Never hardcode API keys in source code
- Use environment variables in production
- Implement key rotation policies

### 3. Model Poisoning

**Mitigation**:
- Download models from trusted sources only
- Verify model checksums
- Use official HuggingFace repositories
- Review model cards and documentation

### 4. Dependency Vulnerabilities

**Mitigation**:
- Regular dependency updates
- Security scanning with `pip-audit`
- Review security advisories
- Pin versions in production

## Secure Development

### Code Review

- All code changes require review
- Security-focused review for sensitive areas
- Automated security scanning in CI/CD

### Testing

- Include security tests
- Test input validation
- Test error handling
- Test authentication/authorization

### Deployment

- Use HTTPS for all API communications
- Implement rate limiting
- Use secure credential storage
- Enable logging and monitoring

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1) and announced via:

- GitHub Security Advisories
- Release notes
- CHANGELOG.md

Subscribe to repository notifications to stay informed.

## Acknowledgments

We appreciate the security research community and thank those who responsibly disclose vulnerabilities.

---

**Last Updated**: October 15, 2025
**Contact**: security@yourdomain.com (replace with your actual email)
