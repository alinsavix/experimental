PORT ?= 4173
NODE ?= node

.PHONY: help build test check clean dist serve run run-packed run-separate run-cdn run-tests vendor-check vendor-update

help:
	@echo "bbrender targets:"
	@echo "  make build          Build packed, separate, and CDN artifacts"
	@echo "  make test           Run automated Node tests"
	@echo "  make check          Run tests and build all artifacts"
	@echo "  make clean          Remove dist output"
	@echo "  make serve          Serve the source project at http://127.0.0.1:$(PORT)/"
	@echo "  make run            Alias for make serve"
	@echo "  make run-tests      Serve source project, then open /tests/visual-harness.html manually"
	@echo "  make run-packed     Serve dist/packed"
	@echo "  make run-separate   Serve dist/separate"
	@echo "  make run-cdn        Serve dist/cdn"
	@echo "  make vendor-check   Check vendored Streamer.bot client version"
	@echo "  make vendor-update  Update vendored Streamer.bot client to latest"

build:
	$(NODE) scripts/build.mjs

test:
	$(NODE) scripts/test.mjs

check:
	$(NODE) scripts/check.mjs

dist: build

clean:
	$(NODE) -e "require('fs').rmSync('dist', { recursive: true, force: true })"

serve:
	$(NODE) scripts/serve.mjs . $(PORT)

run: serve

run-tests:
	@echo "Serving source project. Open http://127.0.0.1:$(PORT)/tests/visual-harness.html"
	$(NODE) scripts/serve.mjs . $(PORT)

run-packed: build
	$(NODE) scripts/serve.mjs dist/packed $(PORT)

run-separate: build
	$(NODE) scripts/serve.mjs dist/separate $(PORT)

run-cdn: build
	$(NODE) scripts/serve.mjs dist/cdn $(PORT)

vendor-check:
	$(NODE) scripts/vendor-streamerbot-client.mjs check

vendor-update:
	$(NODE) scripts/vendor-streamerbot-client.mjs update
