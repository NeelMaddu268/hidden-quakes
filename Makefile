SHELL := /bin/bash
RUN ?=

.PHONY: help check check-py check-web contracts check-contracts publish-run fetch-run runs

help:
	@echo "make check                  typecheck + lint + smoke tests (run before every PR)"
	@echo "make contracts              regenerate TS from the Python contracts"
	@echo "make check-contracts        regenerate and fail if the committed TS/schema differ"
	@echo "make publish-run RUN=<id>   share a run's tables with the team (GitHub release)"
	@echo "make fetch-run RUN=<id>     download a teammate's run tables"
	@echo "make runs                   list shared runs"

check: check-py check-web

check-py:
	@cd services/seismic && uv run ruff check . ../../packages/contracts/python && { uv run pytest -q -m smoke; code=$$?; [ $$code -eq 0 ] || [ $$code -eq 5 ]; }

check-web:
	@if [ ! -d node_modules ]; then echo "check-web: run 'pnpm install' at the repo root first"; exit 1; fi; \
	pnpm -r --if-present typecheck && pnpm -r --if-present lint && pnpm -r --if-present test

contracts:
	bash scripts/gen-contracts.sh

check-contracts: contracts
	@git diff --exit-code -- packages/contracts && echo "contracts up to date"

publish-run:
	@set -e; \
	test -n "$(RUN)" || { echo "usage: make publish-run RUN=<runId>"; exit 1; }; \
	test -d "data/showcase/runs/$(RUN)" || { echo "no such run: data/showcase/runs/$(RUN)"; exit 1; }; \
	tarball="$${TMPDIR:-/tmp}/run-$(RUN).tgz"; \
	tar -czf "$$tarball" -C data/showcase/runs "$(RUN)"; \
	if gh release view "run-$(RUN)" >/dev/null 2>&1; then \
	  gh release upload "run-$(RUN)" "$$tarball" --clobber; \
	else \
	  gh release create "run-$(RUN)" "$$tarball" --prerelease --title "run $(RUN)" --notes "Pipeline run tables. Fetch with: make fetch-run RUN=$(RUN)"; \
	fi; \
	echo "published run-$(RUN)"

fetch-run:
	@set -e; \
	test -n "$(RUN)" || { echo "usage: make fetch-run RUN=<runId>"; exit 1; }; \
	mkdir -p data/showcase/runs; \
	tmp="$$(mktemp -d)"; \
	gh release download "run-$(RUN)" -p "run-$(RUN).tgz" -D "$$tmp" --clobber; \
	tar -xzf "$$tmp/run-$(RUN).tgz" -C data/showcase/runs; \
	echo "fetched into data/showcase/runs/$(RUN)"

runs:
	@gh release list --limit 50 | grep '^run ' || echo "no shared runs yet"
