# Security and account migration

## Required incident response

The public Git history previously contained application/database credentials and a default plaintext doctor account. **Treat those values as compromised.** Rotate the database password and Flask signing secret in the actual environment; updating Git cannot revoke them. Review database access, invalidate existing browser sessions, and assess whether exposed records require notification or other incident response. Removing values from the current branch does not remove them from historical commits.

## Legacy accounts

Old registrations saved plaintext passwords. The updated application intentionally does **not** accept those legacy records: existing users must have passwords reset through a trusted administrative process. Do not export or redistribute their existing passwords. Never run an automatic conversion of plaintext data on an untrusted dataset without evaluating the exposure risk. New registrations use Werkzeug's salted password hashing. The users.password column can hold the resulting hashes (VARCHAR(255)).

Before enabling this app publicly, implement a verified account recovery/reset workflow or securely re-provision users and disable old credentials.

## Application security limitations

- This branch does not provide CSRF protection or rate limiting yet; do not expose it publicly until these are implemented and verified.
- New browser sessions are marked Secure by default; for localhost HTTP-only development, set COOKIE_SECURE=false.
- Medical uploads and predictions may be sensitive. Configure retention/access policies and avoid logging private details.
- Upload validation checks image type and size, but a dedicated image processing sandbox, malware checks, and content security controls may be needed for production.
- The old model artifact and model quality have **not** been evaluated in this branch.
- No security or integration tests have been executed in a live MySQL/model environment.
