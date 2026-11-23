# RQ1 Phase 1B evidence transfer package

**Purpose:** move the verified Phase 1B exposure-reconciliation evidence from the laptop to the university workstation (DESKTOP-AIPC-02) for the scientific-preparation tooling on branch `sci/rq1-phase1-task-oracle-prep`.

**Status:** preparation evidence only. Not an approved manifest, roster, label set or scientific result. Nothing in this folder is code, and nothing here changes any experiment.

| File | Purpose |
|---|---|
| `phase1b-20261123.zip` | The package: 30 files, 2,523,087 bytes uncompressed, single root folder `phase1b-20261123/` |
| `phase1b-20261123.zip.sha256` | External checksum, in `sha256sum` format |

ZIP SHA-256: `9f1ad5a061766fcdd61888dd3073738d300d2acb97627c05e3c0fdf8526b6f64`

## What the archive contains

- The original bytes of the frozen task manifests:
  - acquisition-180: `d002ec6f73bc35a146ffbf1991416c048e2ef7a6cb668051e94348f5dd229219`;
  - acquisition-extension-60: `860c66eedd7cd3c99761682b9d21df1d68cc5cd6c9c1fb8ccce194c1b0ded3a0`;
  - evaluation-amended: `340f52ce660f703d9b98e1cf2a58ac195e2fe7c1c380f08b98d0419929bdb429`.
- 21 small non-scientific prelaunch run records (plans and reports from September 2026), taken from the 2026-09-14 experiment archive.
- Two committed repository files at `275ae94`.
- Package metadata:
  - `SHA256SUMS`, with 29 entries covering every other file in the package;
  - `package-inventory.json`, recording provenance, checks and line endings;
  - `README.md`, the full provenance report;
  - `exposure-sources.phase1b-candidate.json`, a **suggested** exposure config. It is not applied, and applying it is a research decision.

No file was normalised or re-serialised.

## Verify and extract (Linux)

```sh
sha256sum -c phase1b-20261123.zip.sha256                   # must print: phase1b-20261123.zip: OK
python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; \
assert all(n.startswith('phase1b-20261123/') and '..' not in n.split('/') for n in z.namelist()); print(len(z.namelist()),'entries OK')" phase1b-20261123.zip
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" phase1b-20261123.zip /home/researcher
cd /home/researcher/phase1b-20261123 && sha256sum -c SHA256SUMS   # 29 x OK
```

Use the extracted `evidence/` directory as `--sources-base` for `python -m rq1.scientific_prep exposure`. See the package `README.md`, section 4.
