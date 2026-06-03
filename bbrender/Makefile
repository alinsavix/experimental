PORT ?= 4173
NODE ?= node

# Stamp marking that Playwright's Chromium has been installed, so setup only
# pays the install cost once instead of on every test run.
PW_STAMP := node_modules/.playwright-chromium-installed

.PHONY: help build test browser-test visual-test visual-update check clean dist serve run run-packed run-separate run-cdn run-tests setup install-browsers vendor-check vendor-update

help:
	@echo "bbrender targets:"
	@echo "  make build          Build packed, separate, and CDN artifacts"
	@echo "  make test           Run automated Node tests"
	@echo "  make browser-test   Run automated Playwright browser tests"
	@echo "  make visual-test    Run Playwright visual screenshot regression tests"
	@echo "  make visual-update  Update Playwright visual screenshot baselines"
	@echo "  make check          Run tests and build all artifacts"
	@echo "  make clean          Remove dist output"
	@echo "  make serve          Serve the source project at http://127.0.0.1:$(PORT)/"
	@echo "  make run            Alias for make serve"
	@echo "  make run-tests      Serve source project, then open /tests/visual-harness.html manually"
	@echo "  make setup          Install npm deps and Playwright Chromium"
	@echo "  make install-browsers Force-(re)install Playwright Chromium"
	@echo "  make run-packed     Serve dist/packed"
	@echo "  make run-separate   Serve dist/separate"
	@echo "  make run-cdn        Serve dist/cdn"
	@echo "  make vendor-check   Check vendored Streamer.bot client version"
	@echo "  make vendor-update  Update vendored Streamer.bot client to latest"

build:
	$(NODE) scripts/build.mjs

test:
	$(NODE) scripts/test.mjs

browser-test: setup
	npx playwright test

visual-test: setup
	npx playwright test tests/visual.spec.js

visual-update: setup
	npx playwright test tests/visual.spec.js --update-snapshots

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

# Ensure npm deps and the Playwright browser are present before running tests.
setup: node_modules $(PW_STAMP)

# Install npm deps whenever package.json is newer than the install. Touching the
# directory afterwards refreshes its mtime so make sees the install as current.
node_modules: package.json
	npm install
	@touch node_modules

$(PW_STAMP): node_modules
	npx playwright install chromium
	@touch $(PW_STAMP)

# Force a (re)install of the Playwright browser regardless of the stamp.
install-browsers:
	npx playwright install chromium
	@touch $(PW_STAMP)

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
