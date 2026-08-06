.PHONY: help preflight doctor status setup setup-dry-run verify-installation setup-status validate mock test hermes-capabilities verify-hermes-fake verify-hermes-real profiles-plan profiles-create-base profiles-validate profiles-isolation-test profiles-contamination-check recovery-plan recovery-verify recovery-capabilities pilot-list pilot-plan pilot-fake pilot-real pilot-status pilot-report freeze-plan acquisition-plan snapshots-plan evaluation-profiles-plan evaluation-activation-plan tasks-capabilities tasks-pilot-plan

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

hermes-capabilities:
	@python -m rq1.cli hermes-capabilities

verify-hermes-fake:
	@python -m rq1.cli verify-hermes-integration --mode fake

verify-hermes-real:
	@RQ1_RUN_REAL_HERMES_TESTS=1 python -m rq1.cli verify-hermes-integration --mode real

profiles-plan:
	@python -m rq1.cli profiles plan

profiles-create-base:
	@python -m rq1.cli profiles create-base --yes

profiles-validate:
	@python -m rq1.cli profiles validate rq1-pilot
	@python -m rq1.cli profiles validate rq1-acquisition

profiles-isolation-test:
	@python -m rq1.cli profiles isolation-test

profiles-contamination-check:
	@python -m rq1.cli profiles contamination-check rq1-pilot

recovery-plan:
	@python -m rq1.cli recovery plan

recovery-verify:
	@python -m rq1.cli recovery verify --mode fake

recovery-capabilities:
	@python -m rq1.cli recovery capabilities

pilot-list:
	@python -m rq1.cli pilot list

pilot-plan:
	@python -m rq1.cli pilot plan --mode real

pilot-fake:
	@python -m rq1.cli pilot run --mode fake

pilot-real:
	@RQ1_RUN_REAL_PILOT_TESTS=1 python -m rq1.cli pilot run --mode real --yes

pilot-status:
	@python -m rq1.cli pilot status

pilot-report:
	@python -m rq1.cli pilot report

freeze-plan:
	@python -m rq1.cli freeze plan

acquisition-plan:
	@python -m rq1.cli acquisition plan

snapshots-plan:
	@python -m rq1.cli snapshots plan

evaluation-profiles-plan:
	@python -m rq1.cli evaluation profiles plan

evaluation-activation-plan:
	@python -m rq1.cli evaluation activation plan

tasks-capabilities:
	@python -m rq1.cli tasks capabilities

tasks-pilot-plan:
	@python -m rq1.cli tasks propose --kind pilot
