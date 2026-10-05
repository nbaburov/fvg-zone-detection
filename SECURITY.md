# Security

## Status

FVG Zone Detection is a finished university project. It receives no feature work and is not deployed anywhere. It is a research tool, not trading advice: the README's evaluation note explains why its best trading result is not evidence of an edge.

## Secrets

No credentials are committed. The data pipeline and the paper-trading harness read `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` from a local `.env`, and [`.env.example`](.env.example) ships with placeholders only. `ALPACA_PAPER` defaults to `true`; setting it to `false` lets the harness place live orders with real money, so leave it alone unless you mean it.

## Reporting a vulnerability

If you find a security problem in the code, please report it privately through [GitHub's private vulnerability reporting](https://github.com/nbaburov/fvg-zone-detection/security/advisories/new) rather than opening a public issue. Expect an acknowledgement within a week.
