# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions of **GloHealth AI**:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

The GloHealth AI team takes security and user data protection seriously, particularly given the medical and personal data context of health symptom reporting.

If you discover a potential security vulnerability, please follow responsible disclosure:

1. **Do not disclose the issue publicly** (e.g., via GitHub issues, pull requests, or social media).
2. Report the vulnerability privately to the maintainer via:
   - **GitHub Security Advisory:** [Report a security vulnerability](https://github.com/itzabd/GloHealth_Ai/security/advisories/new)
   - **Direct Contact:** [Abdullah Hossien (@itzabd)](https://github.com/itzabd)
3. Include the following details in your report:
   - Type of vulnerability (e.g., SQL injection, authentication bypass, CSRF, privilege escalation).
   - Step-by-step instructions or proof-of-concept to reproduce the vulnerability.
   - Any potential impact on users, patients, or data privacy.
   - Recommended remediation if available.

### Response Timeline
- **Initial acknowledgment:** Within 48 hours.
- **Vulnerability assessment & triage:** Within 5 business days.
- **Fix and release:** As quickly as possible based on severity.

---

## Environment Secrets & Credentials
- **Never commit `.env` files** containing live database credentials, JWT secrets, or API keys.
- Always use `.env.example` as a template and provide variables via your hosting platform's secure environment settings (e.g., Render Environment Variables, Supabase Project Settings).
