# Contributing to medical document Q&A

Use the local setup in [README.md](README.md). Start with a small engineering change and a reproducible case; discuss changes to the application's medical scope before implementation.

Useful work includes offline ingestion/retrieval fixtures, citation and abstention evaluation, content-hash tracking, client session isolation, a dependency lock, and tests for provider failures.

A pull request should state:

- The problem and expected behavior.
- The documents or synthetic fixture used, with redistribution terms.
- Commands, environment versions, results, and untested paths.
- Whether a result measures connectivity, retrieval, answer quality, or another property.

Add focused tests for changed behavior. Use mocked providers for offline checks; do not require real tokens or patient information in CI. A successful model response is not evidence of medical accuracy. Do not publish clinical-performance claims without an appropriate reviewed evaluation.

Do not commit PDFs, runtime indexes, chat history, keys, or private health data. See [SECURITY.md](SECURITY.md).

No standalone LICENSE file is present. Identify third-party sources and terms in proposed additions.
