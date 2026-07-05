# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.5.x   | ✅ Yes    |
| 0.4.x   | ⚠️ Critical fixes only |
| < 0.4   | ❌ No     |

## Reporting a Vulnerability

If you discover a security vulnerability in HumaneProxy, **please do not open a public issue**.

Instead, report it through one of these channels:

1. **GitHub Private Vulnerability Reporting:**
   Navigate to the [Security tab](https://github.com/Vishisht16/Humane-Proxy/security) of this repository and click **"Report a vulnerability"**.

2. **Email:**
   Send details to **mishra@vishisht.tech** with the subject line `[SECURITY] HumaneProxy vulnerability report`.

### What to Include

- A description of the vulnerability and its potential impact.
- Steps to reproduce (proof of concept if possible).
- The affected version(s).
- Any suggested fix or mitigation.

### What to Expect

- **Acknowledgement** within 48 hours.
- A **fix timeline** communicated within 7 days.
- Credit in the release notes (unless you prefer to remain anonymous).

### Scope

The following are in-scope for security reports:

- Bypasses of the safety pipeline (e.g., a message that should be flagged as `self_harm` but is classified as `safe`).
- Injection attacks via webhook payloads, admin API, or MCP server.
- Denial-of-service vectors (e.g., ReDoS in keyword matching).
- Information disclosure through the admin API or escalation logs.

### Out of Scope

- Social engineering of LLM providers (OpenAI, Groq) — these are upstream services.
- Vulnerabilities in dependencies that do not directly affect HumaneProxy's behaviour.
- Theoretical attacks that require physical access to the host machine.

## Disclosure Policy

We follow **coordinated disclosure**. Please allow us a reasonable window (typically 30 days) to address the vulnerability before public disclosure. We will work with you on a timeline.

## Logging & Data-Exposure Hardening

As of v0.5.5:

- Webhook logs contain only the scheme and host of webhook URLs — never the path, query, or fragment, which carry routing tokens for Slack/Discord/Teams.
- Webhook error logs record HTTP status and response length only; response bodies are logged exclusively at `DEBUG` level.
- The reverse proxy never returns raw upstream response bodies to clients; non-JSON upstream responses produce a generic error message, with the body available to operators at `DEBUG` level.
