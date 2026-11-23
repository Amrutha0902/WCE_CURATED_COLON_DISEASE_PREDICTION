"""Operator-controlled prediction metadata retention. Dry-run by default.

Never runs during requests or application startup. Requires explicit database
selection and confirmation; does not delete image files or user accounts.
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
import mysql.connector


def retention_cutoff(days, now=None):
    if not 1 <= days <= 36500:
        raise ValueError("Retention days must be between 1 and 36500")
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(days=days)).replace(tzinfo=None)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--database", required=True, help="Exact expected database name")
    parser.add_argument("--execute", action="store_true", help="Actually delete expired prediction metadata")
    args = parser.parse_args()
    cutoff = retention_cutoff(args.days)
    db_name = os.getenv("DB_NAME", "colon_disease_db")
    if args.database != db_name:
        raise SystemExit("Database name mismatch; refusing to proceed")
    if args.execute and os.getenv("WCE_RETENTION_DELETE_CONFIRM") != f"DELETE_EXPIRED_PREDICTIONS_FROM_{db_name}":
        raise SystemExit("Explicit WCE_RETENTION_DELETE_CONFIRM required for deletion")
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "colon_app"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": db_name,
    }
    connection = mysql.connector.connect(**config)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_timestamp < %s", (cutoff,))
            count = cursor.fetchone()[0]
            print(f"Database: {db_name}; cutoff (UTC): {cutoff.isoformat()}; eligible rows: {count}")
            if args.execute:
                cursor.execute("DELETE FROM predictions WHERE prediction_timestamp < %s", (cutoff,))
                connection.commit()
                print(f"Deleted rows: {cursor.rowcount}")
            else:
                print("DRY RUN: no records changed")
        finally:
            cursor.close()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
