---
name: local-critic
description: Reviews a change on a local model. Advisory only.
model_selector:
  provider: ollama
  model: your-model:latest
tools: []
write_authority: none
requires_primary_review: true
privacy_policy: internal
---

An agent definition. The frontmatter is what `agent_gateway` reads.
