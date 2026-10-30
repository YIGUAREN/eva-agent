"""Print an ADMIN_PASSWORD_HASH value for .env."""

from __future__ import annotations

import getpass
import sys

from eva_agent.server.passwords import hash_password


def main() -> int:
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        print("passwords do not match", file=sys.stderr)
        return 1
    try:
        print(hash_password(password))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
