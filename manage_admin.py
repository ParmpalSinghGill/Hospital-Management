"""Manage admin console accounts (stored hashed in admin_users.json).

    python manage_admin.py add <username> [--role admin|viewer]   # prompts for password
    python manage_admin.py passwd <username>
    python manage_admin.py list
    python manage_admin.py remove <username>
    python manage_admin.py hash                                    # print a hash for ADMIN_PASS_HASH
"""

from __future__ import annotations

import argparse
import getpass
import sys

import admin_auth


def _prompt_password() -> str:
    p1 = getpass.getpass("New password (min 10 chars): ")
    p2 = getpass.getpass("Repeat password: ")
    if p1 != p2:
        raise SystemExit("Passwords do not match.")
    if admin_auth.is_weak_password(p1):
        raise SystemExit("Password too weak (min 10 chars, not a common password).")
    return p1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Admin account management")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("username"); a.add_argument("--role", default="admin", choices=admin_auth.ROLES)
    p = sub.add_parser("passwd"); p.add_argument("username")
    sub.add_parser("list")
    r = sub.add_parser("remove"); r.add_argument("username")
    sub.add_parser("hash")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        users = admin_auth.list_users()
        if not users:
            print("(no accounts in admin_users.json)")
        for u in users:
            print(f"{u.username}\t{u.role}")
        return
    if args.cmd == "remove":
        print("removed" if admin_auth.remove_user(args.username) else "not found")
        return
    if args.cmd == "hash":
        print(admin_auth.hash_password(_prompt_password()))
        return
    if args.cmd in ("add", "passwd"):
        existing = {u.username: u for u in admin_auth.list_users()}
        if args.cmd == "passwd" and args.username not in existing:
            raise SystemExit("no such user")
        role = getattr(args, "role", None) or existing.get(args.username, admin_auth.AdminUser(args.username)).role
        admin_auth.upsert_user(args.username, _prompt_password(), role=role)
        print(f"ok: {args.username} ({role}) -> {admin_auth._users_path()}")
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
