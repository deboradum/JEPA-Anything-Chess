PYTHON ?= python3
DESIGN := recipes/synthetic-linear-dynamics/design.expected.json

.PHONY: help install-dev test test-core test-skill validate-design recipe manifest json compile check

help:
	@echo "JEPA Anything repository checks"
	@echo "  make install-dev      Install the core package and test tools"

install-dev:
	$(PYTHON) -m pip install -e './jepa-anything-core[dev]'
