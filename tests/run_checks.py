"""Run unit checks and optionally integration suites in newly created disposable databases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent.parent
UNIT_SUITES = (
    "tests.test_ml.FeatureTests",
    "tests.test_ml.TrainingTests",
)
INTEGRATION_SUITES = (
    ("de_etl_test_", "RUN_ETL_INTEGRATION", "tests.test_etl.PipelineIntegrationTests",
     ("schema.sql", "03_core_and_quarantine.sql", "04_mart_route_card.sql")),
    ("de_ml_test_", "RUN_ML_INTEGRATION", "tests.test_ml.MlExtractionTests", ()),
    ("de_ml_publish_test_", "RUN_ML_PUBLISH_INTEGRATION", "tests.test_predictions.PredictionPublicationTests", ()),
)


def event(name: str, **fields: object) -> None:
    print(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": name, **fields}), flush=True)


def run_suites(names: tuple[str, ...], environment: dict[str, str]) -> None:
    subprocess.run([sys.executable, "-W", "error", "-m", "unittest", "-v", *names],
                   cwd=ROOT, env=environment, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true", help="also create disposable databases; PGUSER needs CREATEDB")
    args = parser.parse_args(argv)
    environment = os.environ.copy()
    for _, flag, _, _ in INTEGRATION_SUITES:
        environment.pop(flag, None)
    try:
        event("unit_checks_started")
        run_suites(UNIT_SUITES, environment)
        if args.integration:
            with psycopg.connect(autocommit=True, connect_timeout=10) as admin:
                for prefix, flag, suite, schemas in INTEGRATION_SUITES:
                    database = prefix + uuid4().hex[:12]
                    # A failed CREATE never enters cleanup, so an existing database is never dropped.
                    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
                    event("integration_database_created", database=database, suite=suite)
                    try:
                        if schemas:
                            with psycopg.connect(dbname=database, autocommit=True, connect_timeout=10) as fixture:
                                for filename in schemas:
                                    fixture.execute((ROOT / "sql" / filename).read_text(encoding="utf-8"))
                        run_suites((suite,), environment | {"PGDATABASE": database, flag: "1"})
                    finally:
                        if not database.startswith(prefix) or len(database) != len(prefix) + 12:
                            raise RuntimeError("unexpected disposable database name")
                        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
                        event("integration_database_removed", database=database)
        event("checks_succeeded", integration=args.integration)
    except Exception as exc:
        event("checks_failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
