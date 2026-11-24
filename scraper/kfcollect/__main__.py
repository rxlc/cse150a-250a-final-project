"""kfcollect command line: run | status | derive | report | selftest | export."""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from .config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kfcollect", description=__doc__)
    ap.add_argument("--config", help="config.env path (default /etc/kfcollect/config.env or $KF_CONFIG)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run the collector (used by kfcollect.service)")
    st = sub.add_parser("status", help="check health, send alerts/summaries (used by the status timer)")
    st.add_argument("--summary", action="store_true", help="send a summary now")
    st.add_argument("--print", action="store_true", help="print the status instead of sending")
    st.add_argument("--test", action="store_true", help="send a test message to the webhook")
    st.add_argument("--note", metavar="TEXT", help="post a one-off note (used by update.sh)")
    dv = sub.add_parser("derive", help="build stop passages, segments, 5-min intervals, vehicle steps")
    dv.add_argument("--days", type=int, default=2)
    dv.add_argument("--full", action="store_true", help="rebuild every service day")
    sub.add_parser("report", help="daily report: derive, post (used by the daily timer)")
    sub.add_parser("selftest", help="check endpoints and database")
    ex = sub.add_parser("export", help="export tables as CSV in a zip")
    ex.add_argument("--out", help="zip path (default: <data dir>/exports/kf-export-<date>.zip)")
    ex.add_argument("--since", help="YYYY-MM-DD (local)")
    ex.add_argument("--tables", default="trip_status,stop_times,trip_schedules,ref_trips,ref_stops,ref_routes,"
                                        "situations,arrivals,stop_passages,segments,intervals_5min,vehicle_steps,"
                                        "trip_events,gaps,polls,shapes")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "run":
        from .collector import Collector
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        Collector(cfg).run()
        return 0

    from .store import connect
    if args.cmd in ("derive", "report"):
        args.lock = _wait_for_derive_lock(cfg)  # held until the process exits
    db = connect(cfg.db_path)

    if args.cmd == "status":
        from . import status
        if args.note:
            return 0 if status.send(cfg, "🔧 kfcollect", args.note) else 1
        if args.test:
            ok = status.send(cfg, "👋 kfcollect webhook test", "If you can read this, Discord alerts work.")
            print("sent" if ok else "FAILED to send")
            return 0 if ok else 1
        if args.print:
            cfg.discord_webhook_url = ""
            s = status.gather(db, cfg)
            status.send(cfg, "Collection status", "", fields=status.summary_fields(s))
            return 0
        status.check(db, cfg, force_summary=args.summary)
        return 0

    if args.cmd == "report":
        _daily_report(db, cfg)
        return 0

    if args.cmd == "derive":
        from . import derive
        for r in derive.run(db, cfg, days=args.days, full=args.full):
            day = datetime.fromtimestamp(r["service_date"] / 1000, cfg.timezone).strftime("%Y-%m-%d")
            print(f"{day}: {r['trips']} trips, {r['passages']} stop passages, {r['segments']} segments")
        return 0

    if args.cmd == "selftest":
        return _selftest(cfg, db)

    if args.cmd == "export":
        return _export(cfg, db, args)
    return 1


def _wait_for_derive_lock(cfg):
    """One derivation at a time: the hourly timer, the daily report and an update's background
    re-derive would otherwise write the same days at once. Waits (doesn't fail) for the other to finish."""
    import fcntl
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    fh = open(cfg.data_dir / "derive.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.info("another derive is running; waiting for it to finish")
        fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _daily_report(db, cfg) -> None:
    from . import derive, status
    results = derive.run(db, cfg, days=2)
    lines = [f"{datetime.fromtimestamp(r['service_date'] / 1000, cfg.timezone):%a %m/%d}: {r['trips']} trips, "
             f"{r['passages']} passages, {r['segments']} segments" for r in results]
    s = status.gather(db, cfg)
    status.send(cfg, f"📅 Daily report · {s['now']:%a %b %d}", "\n".join(lines) or "No trips derived yet.",
                fields=status.summary_fields(s))


def _selftest(cfg, db) -> int:
    from .http import Oba, fetch
    ok = True
    oba = Oba(cfg.oba_base_url, cfg.oba_flavor, cfg.oba_api_key)
    for route in cfg.focus_routes:
        r = oba.get("trips-for-route", route, includeStatus=True, includeSchedule=True)
        n = len(((r.json().get("data") or {}).get("list") or [])) if r.ok else "-"
        print(f"[{'ok' if r.ok else 'FAIL'}] trips-for-route {route}: HTTP {r.status}, {n} trips, {r.elapsed_ms} ms {r.error}")
        ok &= r.ok
    for path in ("gtfs.zip", "gtfs-rt/vehicle-positions.pb"):
        r = fetch(f"{cfg.gtfs_base}/{path}", timeout=60)
        print(f"[{'ok' if r.ok else 'FAIL'}] GTFS {path}: HTTP {r.status}, {len(r.body)} bytes {r.error}")
        ok &= r.ok
    try:
        db.execute("SELECT COUNT(*) FROM polls").fetchone()
        print(f"[ok] database {cfg.db_path}")
    except sqlite3.Error as e:
        print(f"[FAIL] database: {e}")
        ok = False
    print(f"[{'ok' if cfg.discord_webhook_url else 'warn'}] Discord webhook "
          f"{'configured' if cfg.discord_webhook_url else 'not set (status prints to stdout)'}")
    return 0 if ok else 1


def _export(cfg, db, args) -> int:
    since = 0
    if args.since:
        since = int(datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=cfg.timezone).timestamp() * 1000)
    time_cols = {"trip_status": "server_time", "arrivals": "server_time", "polls": "requested_at",
                 "stop_passages": "actual_time", "segments": "t_to", "intervals_5min": "slot_start",
                 "vehicle_steps": "t", "trip_events": "t", "gaps": "start_ms", "trip_schedules": "first_seen",
                 "stop_times": "service_date"}
    out = Path(args.out) if args.out else cfg.data_dir / "exports" / f"kf-export-{datetime.now():%Y%m%d-%H%M}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
            col = time_cols.get(table)
            try:
                cur = db.execute(f"SELECT * FROM {table}" + (f" WHERE {col} >= ?" if col and since else ""),
                                 (since,) if col and since else ())
            except sqlite3.OperationalError as e:
                print(f"skip {table}: {e}")
                continue
            with z.open(f"{table}.csv", "w") as fh:
                text = io.TextIOWrapper(fh, "utf-8", newline="")
                w = csv.writer(text)
                w.writerow([d[0] for d in cur.description])
                n = 0
                for row in cur:
                    w.writerow(row)
                    n += 1
                text.flush()
                text.detach()  # leave fh for the with-block to close
            print(f"{table}: {n:,} rows")
    out.chmod(0o644)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
