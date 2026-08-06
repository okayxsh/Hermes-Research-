.PHONY: help preflight doctor status setup setup-dry-run verify-installation setup-status validate mock test

help:
	@python -m rq1.cli --help

preflight:
	@python -m rq1.cli preflight

doctor:
	@python -m rq1.cli doctor

status:
	@python -m rq1.cli stage-status

setup:
	@bash scripts/setup_machine.sh --yes

setup-dry-run:
	@bash scripts/setup_machine.sh --dry-run --verbose

verify-installation:
	@bash scripts/09_verify_installation.sh --yes --resume

setup-status:
	@python -m rq1.cli setup-status

validate:
	@python -m rq1.cli validate-config

mock:
	@python -m rq1.cli mock-run

test:
	@python -m unittest discover -s tests -v
