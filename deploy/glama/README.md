# Glama Marketplace Container

This Dockerfile is **not** how HumaneProxy is served. It exists solely as the
container definition for the [Glama MCP marketplace](https://glama.ai/mcp/servers/Vishisht16/Humane-Proxy)
listing.

If you ever need to build it locally, use the repository root as the build
context (the Dockerfile does `COPY . .`):

```bash
docker build -f deploy/glama/Dockerfile .
```

Note: the `.dockerignore` in this directory only takes effect if copied to the
build-context root first — Docker reads `.dockerignore` from the context, not
from the Dockerfile's location.
