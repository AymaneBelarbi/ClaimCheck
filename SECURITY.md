# Security

ClaimCheck ships only synthetic data, and it is not deployed anywhere. Still, if you find a way around its access controls, please report it privately rather than in a public issue.

## Reporting

Use GitHub's private vulnerability reporting: the **Security** tab of this repository, then **Report a vulnerability**. Include the steps to reproduce and what you could read or change. I aim to reply within a week.

## In scope

- a customer token reading or changing another customer's claim, through the API or around it (row-level security in `db/migrations/`)
- the adjuster agent reaching a tool outside the MCP allowlist, acting on a claim it was not asked about, or writing without human confirmation
- names or ID numbers reaching a cloud LLM backend in a way the documented limits in [ADR-006](docs/adr/0006-llm-gateway-and-redaction.md) do not already cover
- document text, prompts or model output ending up in traces
- bypassing the upload size and page limits

## Out of scope

The dev secrets in `docker-compose.yml` and `.env.example` are placeholders; the API logs a warning when it starts with one. Missing TLS, rate limiting or a production identity provider are known gaps of a local demo, not vulnerabilities.
