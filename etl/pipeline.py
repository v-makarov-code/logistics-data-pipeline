"""Run the source-to-core batch. All data writes share one transaction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import psycopg

SQL_DIR = Path(__file__).parent / "sql"
MART_SQL = Path(__file__).parent.parent / "sql" / "05_refresh_route_card.sql"
MART_START = "-- BEGIN SHARED MART REFRESH"
MART_END = "-- END SHARED MART REFRESH"
LOGGER = logging.getLogger("etl")
# A single session-level lock serializes all runs, including date moves across windows.
LOCK_ID = 2026111101


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use YYYY-MM-DD") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reprocess prices for [start-date, end-date).")
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--as-of-date", type=parse_date, default=date(2026, 10, 26),
                        help="mart snapshot date, defaults to 2026-10-26")
    args = parser.parse_args(argv)
    if args.start_date >= args.end_date:
        parser.error("--start-date must be earlier than --end-date")
    return args


def log_event(event: str, **fields: object) -> None:
    LOGGER.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "event": event, **fields}, ensure_ascii=False, default=str))


def execute_file(cur: psycopg.Cursor, name: str, params: dict | None = None) -> None:
    query = (SQL_DIR / name).read_text(encoding="utf-8")
    cur.execute(query, params)


def mart_refresh_sql() -> str:
    """Reuse the exact calculation from the standalone psql script."""
    script = MART_SQL.read_text(encoding="utf-8")
    if script.count(MART_START) != 1 or script.count(MART_END) != 1:
        raise ValueError("route-card SQL must contain exactly one pair of refresh markers")
    before, _, remaining = script.partition(MART_START)
    body, end, _ = remaining.partition(MART_END)
    if not end or MART_END in before:
        raise ValueError("route-card SQL refresh markers are out of order")
    return body


def process_window(conn: psycopg.Connection, start_date: date, end_date: date, run_id: str,
                   as_of_date: date | None = None) -> dict:
    """Atomically replace Core and, when requested, refresh the complete Mart."""
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")

    # Acquire before opening the snapshot: an overlapping run must never read old state.
    if not conn.autocommit:
        raise ValueError("process_window requires an autocommit connection")
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_ID,))
        if not cur.fetchone()[0]:
            raise RuntimeError("another ETL run is already in progress")
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                execute_file(cur, "01_candidates.sql", {"start_date": start_date, "end_date": end_date})
                cur.execute("CREATE UNIQUE INDEX ON candidate_ids (source_price_id)")
                cur.execute("ANALYZE candidate_ids")
                execute_file(cur, "02_evaluate.sql")
                execute_file(cur, "03_valid.sql")
                # Keep the first detection time while a record remains in quarantine.
                cur.execute("""
                    CREATE TEMP TABLE rejected_prices ON COMMIT DROP AS
                    SELECT e.id AS source_price_id, e.loading_date, e.reason_codes, e.source_data,
                           COALESCE(q.detected_at, now()) AS detected_at
                    FROM evaluated_prices e
                    LEFT JOIN dq.quarantine_prices q ON q.source_price_id = e.id
                    WHERE cardinality(e.reason_codes) > 0
                """)
                cur.execute("""
                    SELECT (SELECT count(*) FROM candidate_ids),
                           (SELECT count(*) FROM evaluated_prices),
                           (SELECT count(*) FROM valid_prices),
                           (SELECT count(*) FROM rejected_prices)
                """)
                candidates, found, valid, rejected = cur.fetchone()
                if found != valid + rejected:
                    raise RuntimeError("validation balance failed; refusing to change target tables")
                cur.execute("""
                    SELECT reason, count(*) FROM rejected_prices,
                    LATERAL unnest(reason_codes) AS reason GROUP BY reason ORDER BY reason
                """)
                reasons = dict(cur.fetchall())
                cur.execute("DELETE FROM core.prices WHERE source_price_id IN (SELECT source_price_id FROM candidate_ids)")
                cur.execute("DELETE FROM dq.quarantine_prices WHERE source_price_id IN (SELECT source_price_id FROM candidate_ids)")
                cur.execute("""
                    INSERT INTO core.prices (
                        source_price_id, source_created_at, source_updated_at, subscriber_id,
                        organization_id, source, loading_date, end_loading_date,
                        loading_city_gcid, unloading_city_gcid, distance, total_weight, total_volume,
                        body_type, weight_category, volume_category, price_original, currency_code,
                        nearest_rate_date, exchange_rate, price_rub, price_type, is_fact, complex, outlier_mark
                    ) SELECT * FROM valid_prices
                """)
                cur.execute("""
                    INSERT INTO dq.quarantine_prices
                        (source_price_id, loading_date, reason_codes, source_data, detected_at, last_run_id)
                    SELECT source_price_id, loading_date, reason_codes, source_data, detected_at, %s::uuid
                    FROM rejected_prices
                """, (run_id,))
                mart_rows = None
                if as_of_date is not None:
                    # Both stages share the same transaction and source snapshot: a failed
                    # Mart calculation rolls the Core and quarantine changes back as well.
                    log_event("core_prepared", run_id=run_id, valid=valid, rejected=rejected,
                              note="not committed until mart succeeds")
                    # On a fresh database Core was empty when PostgreSQL collected stats.
                    # Analyze after the load so Mart sees an accurate row-count estimate.
                    cur.execute("ANALYZE core.prices")
                    log_event("mart_started", run_id=run_id, as_of_date=as_of_date)
                    cur.execute("CREATE TEMP TABLE route_parameters ON COMMIT DROP AS SELECT %s::date AS as_of_date",
                                (as_of_date,))
                    cur.execute(mart_refresh_sql())
                    cur.execute("SELECT count(*) FROM route_card_result")
                    mart_rows = cur.fetchone()[0]
        return {"candidates": candidates, "found": found, "deleted_at_source": candidates - found,
                "valid": valid, "rejected": rejected, "reasons": reasons,
                "mart_rows": mart_rows, "as_of_date": as_of_date}
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_id = str(uuid4())
    log_event("etl_started", run_id=run_id, start_date=args.start_date, end_date=args.end_date,
              as_of_date=args.as_of_date)
    try:
        # PGHOST, PGPORT, PGDATABASE, PGUSER and PGPASSWORD follow libpq conventions.
        with psycopg.connect(autocommit=True, connect_timeout=10) as conn:
            result = process_window(conn, args.start_date, args.end_date, run_id, args.as_of_date)
    except Exception as exc:
        log_event("etl_failed", run_id=run_id, error_type=type(exc).__name__, error=str(exc))
        return 1
    log_event("etl_succeeded", run_id=run_id, **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
