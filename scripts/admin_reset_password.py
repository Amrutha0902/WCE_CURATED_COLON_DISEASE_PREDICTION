"""Administrative password reset for an existing account, run only in a trusted shell.

This does NOT send reset links, validate doctor identity, or rotate leaked DB keys.
The operator must verify the account holder out of band before executing.
"""
import argparse
import getpass
import os

import mysql.connector
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Trusted administrator-only password reset")
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    if not args.username or len(args.username) > 50:
        parser.error("Invalid username")
    if os.environ.get("WCE_ADMIN_RESET_CONFIRM") != "I_VERIFIED_IDENTITY":
        parser.error("Set WCE_ADMIN_RESET_CONFIRM=I_VERIFIED_IDENTITY only after verifying the account holder")
    password = getpass.getpass("New password (not echoed): ")
    confirmation = getpass.getpass("Confirm new password: ")
    if password != confirmation or len(password) < 12:
        parser.error("Passwords do not match or are shorter than 12 characters")
    cfg = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "user": os.environ.get("DB_USER", "colon_app"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", "colon_disease_db"),
    }
    print(f"Target database: {cfg['database']}")
    print(f"Database user: {cfg['user']}")

    confirmation = input(
        f"Type {cfg['database']} to confirm the target database: "
    )

    if confirmation != cfg["database"]:
        raise SystemExit("Database confirmation failed. No changes made.")

    connection = mysql.connector.connect(**cfg)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE users SET password=%s WHERE username=%s AND role='doctor'",
            (generate_password_hash(password), args.username),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise SystemExit("Account not found or not uniquely identified; no change committed")
        connection.commit()
        print("Password reset committed for the specified account. Notify the account holder securely.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
