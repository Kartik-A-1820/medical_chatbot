# Security and application scope

This repository is an educational document-Q&A prototype. It has no clinical validation and must not be used for diagnosis, treatment decisions, or prescribing. The legacy prescription-style PDF workflow is unvalidated.

Keep the API local. It has permissive CORS, no application authentication, process-local history, and incomplete client session separation. Keyword-based emergency checks do not establish reliable triage.

Questions, retrieved document passages, and graph-extraction inputs can be transmitted to configured providers. Use public or synthetic documents. Do not upload patient records, credentials, or private health information.

Indexes and trackers are runtime state and should not be committed. Removing an artifact in a new commit does not remove earlier copies from git history.

For a vulnerability, contact [anagawadi.kartik@gmail.com](mailto:anagawadi.kartik@gmail.com) privately with a redacted reproduction. Do not include real patient data or working keys. Rotate any exposed credential.
