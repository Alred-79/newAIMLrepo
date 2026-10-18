# ─────────────────────────────────────────────────────────────
#  Makefile: standard targets for local development
# ─────────────────────────────────────────────────────────────

.PHONY: help install lint fmt test build clean

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make install   → install dependencies"
	@echo "  make lint      → run linters (JS: ESLint, Terraform fmt)"
	@echo "  make fmt       → auto-format (Prettier, terraform fmt)"
	@echo "  make test      → run tests / smoke tests"
	@echo "  make build     → build artifacts (JS: bundle, Docker)"
	@echo "  make clean     → remove build artifacts / caches"
	@echo ""

install:
ifeq ($(wildcard package.json),package.json)
	@echo "Installing Node.js deps…"
	npm ci
else ifeq ($(wildcard requirements.txt),requirements.txt)
	@echo "Installing Python deps…"
	pip install -r requirements.txt
else
	@echo "Nothing to install here."
endif

lint:
ifeq ($(wildcard package.json),package.json)
	@echo "Running ESLint…"
	npm run lint
else ifeq ($(wildcard terraform),terraform)
	@echo "Running terraform fmt -check…"
	cd terraform && terraform fmt -check
else
	@echo "No lint step defined for this repo."
endif

fmt:
ifeq ($(wildcard package.json),package.json)
	@echo "Running Prettier…"
	npm run fmt
else ifeq ($(wildcard terraform),terraform)
	@echo "Running terraform fmt…"
	cd terraform && terraform fmt
else
	@echo "No fmt step defined for this repo."
endif

test:
ifeq ($(wildcard jest.config.js),jest.config.js)
	@echo "Running Jest tests…"
	npm test -- --watchAll=false
endif
ifeq ($(wildcard tests),tests)
	@echo "Running Python smoke tests…"
	pytest --maxfail=1 --disable-warnings -q
endif

build:
ifeq ($(wildcard package.json),package.json)
	@echo "Building…"
	npm run build
else ifeq ($(wildcard Dockerfile),Dockerfile)
	@echo "Building Docker image…"
	docker build -t bias-ai/$(notdir $(CURDIR)):latest .
else
	@echo "No build step defined for this repo."
endif

clean:
	@echo "Cleaning…"
	rm -rf node_modules dist build __pycache__ .pytest_cache
	cd terraform && rm -rf .terraform
