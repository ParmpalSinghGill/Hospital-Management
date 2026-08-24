"""Encrypted SQLite backups + restore (HIPAA §164.308(a)(7) contingency plan).

* ``create_backup()`` uses the SQLite online-backup API (consistent snapshot),
  then encrypts with Fernet (AES-128-CBC + HMAC) when ``BACKUP_ENC_KEY`` is set
  (generate: ``python backup.py keygen``). Files are written 0600 under
  ``backups/`` (or ``$DATA_DIR/backups``).
* ``restore_backup(path, target)`` decrypts (if needed) and verifies
  ``PRAGMA integrity_check`` before replacing the live DB.
* ``start_scheduler()`` takes a daily backup; pair with off-host copy
  (rsync/object storage) in your ops runbook.

CLI: ``python backup.py create | list | verify <file> | restore <file> | keygen``
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import security_config as sc

KEEP_LAST = int(os.getenv("BACKUP_KEEP_LAST", "14"))


def _key() -> bytes | None:
    raw = (os.getenv("BACKUP_ENC_KEY") or "").strip()
    return raw.encode() if raw else None


def _fernet():
    key = _key()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("BACKUP_ENC_KEY set but 'cryptography' is not installed") from exc
    return Fernet(key)


def db_path() -> Path:
    try:
        import db_core

        return Path(db_core.DB_PATH)
    except Exception:  # noqa: BLE001
        return sc.dataset_dir() / "hospital.db"


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def create_backup(*, dest_dir: Path | None = None, source: Path | None = None) -> Path:
    src = source or db_path()
    if not src.exists():
        raise FileNotFoundError(f"database not found: {src}")
    out_dir = dest_dir or sc.backups_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.chmod(0o700)
    except OSError:
        pass
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory() as tmpd:
        snap = Path(tmpd) / "snapshot.db"
        with sqlite3.connect(str(src)) as con, sqlite3.connect(str(snap)) as dst:
            con.backup(dst)
        data = snap.read_bytes()
    f = _fernet()
    if f:
        target = out_dir / f"hospital-{stamp}.db.enc"
        _write_private(target, f.encrypt(data))
    else:
        target = out_dir / f"hospital-{stamp}.db"
        _write_private(target, data)
    _rotate(out_dir)
    try:
        from audit_log import audit

        audit("backup.create", actor="scheduler", detail={"file": target.name, "encrypted": bool(f), "bytes": len(data)})
    except Exception:  # noqa: BLE001
        pass
    return target


def _rotate(out_dir: Path) -> None:
    files = sorted(
        [p for p in out_dir.iterdir() if p.is_file() and p.name.startswith("hospital-")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in files[KEEP_LAST:]:
        try:
            p.unlink()
        except OSError:
            pass


def list_backups(out_dir: Path | None = None) -> list[Path]:
    d = out_dir or sc.backups_dir()
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.name.startswith("hospital-")], reverse=True)


def _decrypt_to_temp(path: Path) -> Path:
    data = path.read_bytes()
    if path.suffix == ".enc":
        f = _fernet()
        if not f:
            raise RuntimeError("backup is encrypted; set BACKUP_ENC_KEY to restore")
        data = f.decrypt(data)
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    _write_private(tmp, data)
    return tmp


def verify_backup(path: Path) -> bool:
    tmp = _decrypt_to_temp(path)
    try:
        with sqlite3.connect(str(tmp)) as con:
            row = con.execute("PRAGMA integrity_check").fetchone()
            return bool(row and row[0] == "ok")
    finally:
        tmp.unlink(missing_ok=True)


def restore_backup(path: Path, *, target: Path | None = None) -> Path:
    dst = target or db_path()
    tmp = _decrypt_to_temp(path)
    try:
        with sqlite3.connect(str(tmp)) as con:
            row = con.execute("PRAGMA integrity_check").fetchone()
            if not row or row[0] != "ok":
                raise RuntimeError("backup failed integrity check; not restored")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            keep = dst.with_name(dst.name + f".pre-restore-{int(time.time())}")
            shutil.copy2(dst, keep)
            try:
                keep.chmod(0o600)
            except OSError:
                pass
        shutil.copy2(tmp, dst)
        try:
            dst.chmod(0o600)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)
    try:
        from audit_log import audit

        audit("backup.restore", actor="operator", detail={"file": path.name})
    except Exception:  # noqa: BLE001
        pass
    return dst


_THREAD: threading.Thread | None = None


def start_scheduler(interval_secs: int = 24 * 3600, *, initial_delay_secs: int = 120) -> None:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return

    def _loop() -> None:
        time.sleep(initial_delay_secs)
        while True:
            try:
                create_backup()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval_secs)

    _THREAD = threading.Thread(target=_loop, name="backup-scheduler", daemon=True)
    _THREAD.start()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Encrypted SQLite backups.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create")
    sub.add_parser("list")
    sub.add_parser("keygen")
    v = sub.add_parser("verify"); v.add_argument("file")
    r = sub.add_parser("restore"); r.add_argument("file")
    args = parser.parse_args(argv)
    if args.cmd == "keygen":
        from cryptography.fernet import Fernet

        print(Fernet.generate_key().decode())
        return
    if args.cmd == "create":
        print(create_backup())
        return
    if args.cmd == "list":
        for p in list_backups():
            print(p)
        return
    if args.cmd == "verify":
        ok = verify_backup(Path(args.file))
        print("ok" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    if args.cmd == "restore":
        print(restore_backup(Path(args.file)))


if __name__ == "__main__":
    main()
