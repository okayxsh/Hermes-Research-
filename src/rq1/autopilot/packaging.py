from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
from typing import Iterable

EXCLUDED_NAMES=(".env","credentials","token","secret","password")
REQUIRED=("FINAL_SUMMARY.md","runtime_summary.json","metrics.json","checksums.sha256")
def _allowed(path: Path) -> bool: return not any(word in path.name.lower() for word in EXCLUDED_NAMES)
def sha(path: Path) -> str:
    digest=hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def write_checksums(root: Path) -> Path:
    rows=[f"{sha(path)}  {path.relative_to(root).as_posix()}" for path in sorted(root.rglob("*")) if path.is_file() and path.name!="checksums.sha256" and _allowed(path)]
    path=root/"checksums.sha256"; path.write_text("\n".join(rows)+"\n",encoding="utf-8"); return path
def package(root: Path) -> tuple[Path,dict]:
    checksums=write_checksums(root); archive=root/"rq1-final-reproducibility-package.zip"
    if archive.exists(): raise FileExistsError("refusing to overwrite final archive")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != archive and _allowed(path): zip_file.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as zip_file:
        bad=zip_file.testzip(); names=set(zip_file.namelist())
    missing=[name for name in REQUIRED if name not in names]
    report={"valid":bad is None and not missing,"archive":str(archive),"archive_sha256":sha(archive),"archive_size":archive.stat().st_size,"missing":missing,"bad_member":bad}
    (root/"archive-validation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return archive,report
