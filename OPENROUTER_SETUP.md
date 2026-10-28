# OpenRouter configuration

Set `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and an `OPENROUTER_MODEL` available to your account in `backend/.env`. Disable OmniRoute when you do not intend to use a local gateway.

If a request reports “No endpoints found matching your data policy,” inspect the selected model and your account's [privacy settings](https://openrouter.ai/settings/privacy). Read the terms and data-handling requirements before changing a setting. Do not relax privacy controls simply to make a demo work.

Model IDs, pricing, routing availability, and provider policies can change. This repository does not guarantee a free route or a particular model. Use public or synthetic documents for experiments, and record the actual provider/model used.

The provider diagnostic is described in [README.md](README.md). It makes live requests and is not a retrieval-quality or medical-safety test.
