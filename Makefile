.PHONY: help preflight doctor status validate mock test

help:
	@python -m rq1.cli --help

preflight:
	@python -m rq1.cli preflight

doctor:
	@python -m rq1.cli doctor

status:
	@python -m rq1.cli stage-status

validate:
	@python -m rq1.cli validate-config

mock:
	@python -m rq1.cli mock-run

test:
	@python -m unittest discover -s tests -v
