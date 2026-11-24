"""Collection status: alerts and summaries to a Discord webhook (or stdout)."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
import zipfile
from datetime import datetime

from . import __version__
from .config import Config
from .http import fetch
from .store import dir_size_bytes, free_gb, now_ms

GREEN, YELLOW, RED, BLUE = 0x2ECC71, 0xF1C40F, 0xE74C3C, 0x3498DB
REMIND_EVERY_S = 2 * 3600


# --- facts --------------------------------------------------------------------------------

def _scalar(db, sql, args=()):
    try:
        row = db.execute(sql, args).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def service_holidays(cfg: Config) -> set[str]:
    """Weekday-service exception dates (YYYYMMDD) from the newest saved GTFS feed."""
    zips = sorted(cfg.gtfs_dir.glob("gtfs-*.zip"))
    if not zips:
        return set()
    try:
        with zipfile.ZipFile(zips[-1]) as z:
            rows = list(csv.DictReader(io.StringIO(z.read("calendar_dates.txt").decode("utf-8-sig"))))
        return {r["date"] for r in rows if r.get("exception_type") == "2"}
    except Exception:
        return set()


def in_service_hours(cfg: Config, now: datetime) -> bool:
    if now.weekday() >= 5 or now.strftime("%Y%m%d") in service_holidays(cfg):
        return False
    hm = now.strftime("%H:%M")
    return cfg.service_start <= hm <= cfg.service_end


def gather(db: sqlite3.Connection, cfg: Config) -> dict:
    now = datetime.now(cfg.timezone)
    t = now_ms()
    hour_ago = t - 3600_000
    day_start = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    try:
        hb = json.loads(cfg.heartbeat_path.read_text())
    except Exception:
        hb = {}
    by_kind = {}
    for r in db.execute(
            "SELECT kind, SUM(error IS NULL AND http_status BETWEEN 200 AND 299) ok, COUNT(*) n, "
            "AVG(elapsed_ms) ms FROM polls WHERE requested_at >= ? GROUP BY kind", (hour_ago,)):
        by_kind[r["kind"]] = {"ok": r["ok"] or 0, "n": r["n"], "ms": round(r["ms"] or 0)}
    focus = cfg.focus_routes
    marks = ",".join("?" * len(focus))
    per_route = {}
    # Count each bus under its trip's own route (trips-for-route also lists other routes' trips in
    # shared blocks, e.g. Inside/Outside Loop), over the last 2 minutes so a bus that just switched
    # loops isn't counted on both.
    for r in db.execute(
            f"SELECT COALESCE(rt.route_id, ts.route_id) AS route, COUNT(DISTINCT ts.vehicle_id) v, COUNT(*) n, "
            f"AVG(ts.schedule_deviation) dev FROM trip_status ts "
            f"LEFT JOIN (SELECT trip_id, route_id FROM ref_trips WHERE day = (SELECT MAX(day) FROM ref_trips)) rt "
            f"ON rt.trip_id = ts.trip_id "
            f"WHERE ts.server_time >= ? AND ts.predicted = 1 AND ts.vehicle_id IS NOT NULL "
            f"AND (ts.active_trip_id IS NULL OR ts.active_trip_id = ts.trip_id) "
            f"GROUP BY route HAVING route IN ({marks})", (t - 120_000, *focus)):
        per_route[r["route"]] = {"vehicles": r["v"], "rows": r["n"], "avg_dev_s": round(r["dev"] or 0)}
    names = {r[0]: r[1] for r in db.execute(
        "SELECT route_id, COALESCE(NULLIF(long_name,''), short_name) FROM ref_routes "
        "WHERE day = (SELECT MAX(day) FROM ref_routes)")}
    raw_today = cfg.raw_dir / now.strftime("%Y-%m-%d")
    return {
        "now": now, "heartbeat": hb,
        "heartbeat_age_s": (t - hb["ts"]) / 1000 if hb.get("ts") else None,
        "last_ok_age_s": (t - hb["last_ok_poll"]) / 1000 if hb.get("last_ok_poll") else None,
        "polls_hour": by_kind,
        "rows_hour": _scalar(db, "SELECT COUNT(*) FROM trip_status WHERE server_time >= ?", (hour_ago,)) or 0,
        "rows_today": _scalar(db, "SELECT COUNT(*) FROM trip_status WHERE server_time >= ?", (day_start,)) or 0,
        "rows_total": _scalar(db, "SELECT MAX(rowid) FROM trip_status") or 0,
        "arrivals_today": _scalar(db, "SELECT COUNT(*) FROM arrivals WHERE server_time >= ?", (day_start,)) or 0,
        "schedules_today": _scalar(db, "SELECT COUNT(*) FROM trip_schedules WHERE first_seen >= ?", (day_start,)) or 0,
        "passages_today": _scalar(db, "SELECT COUNT(*) FROM stop_passages WHERE actual_time >= ?", (day_start,)),
        "gap_minutes_today": _scalar(db, "SELECT COALESCE(SUM(minutes),0) FROM gaps WHERE start_ms >= ?", (day_start,)),
        "active_alerts": _scalar(db, "SELECT COUNT(*) FROM situations WHERE last_seen >= ?", (t - 600_000,)) or 0,
        "first_data": _scalar(db, "SELECT MIN(requested_at) FROM polls"),
        "focus": {r: {**per_route.get(r, {"vehicles": 0, "rows": 0, "avg_dev_s": None}), "name": names.get(r, r)}
                  for r in focus},
        "db_gb": cfg.db_path.stat().st_size / 1e9 if cfg.db_path.exists() else 0,
        "raw_today_mb": dir_size_bytes(raw_today) / 1e6,
        "raw_total_gb": dir_size_bytes(cfg.raw_dir) / 1e9,
        "free_gb": free_gb(cfg.data_dir) if cfg.data_dir.exists() else None,
        "in_service": in_service_hours(cfg, now),
    }


def _fmt_age(s) -> str:
    if s is None:
        return "never"
    if s < 90:
        return f"{s:.0f}s ago"
    if s < 5400:
        return f"{s / 60:.0f} min ago"
    return f"{s / 3600:.1f} h ago"


# --- Discord ------------------------------------------------------------------------------

EMBED_BUDGET = 5500  # Discord rejects embeds over 6000 characters in total


def _post(cfg: Config, embed: dict) -> bool:
    body = json.dumps({"username": "Killer Fish collector", "embeds": [embed]}).encode()
    for _ in range(3):
        r = fetch(cfg.discord_webhook_url, data=body, headers={"Content-Type": "application/json"}, retries=0)
        if r.status == 429:  # rate limited: wait as told
            try:
                time.sleep(float(r.json().get("retry_after", 2)))
            except Exception:
                time.sleep(2)
            continue
        return r.ok or r.status == 204
    return False


def send(cfg: Config, title: str, description: str = "", color: int = BLUE, fields: list | None = None) -> bool:
    """Post an embed; long field lists are split across several messages to stay within Discord's limits."""
    title, description = title[:256], description[:3000]
    items = [{"name": n[:256], "value": (v or "—")[:1024], "inline": inline} for n, v, inline in fields or []]
    if not cfg.discord_webhook_url:
        print(f"\n== {title}\n{description}")
        for f in items:
            print(f"  {f['name']}: {f['value']}")
        return True
    pages: list[list[dict]] = [[]]
    size = len(title) + len(description)
    for f in items:
        n = len(f["name"]) + len(f["value"])
        if pages[-1] and (size + n > EMBED_BUDGET or len(pages[-1]) >= 25):
            pages.append([])
            size = len(title) + 12
        pages[-1].append(f)
        size += n
    ok = True
    for i, page in enumerate(pages):
        embed = {"title": title if i == 0 else f"{title} (cont. {i + 1}/{len(pages)})", "color": color,
                 "footer": {"text": f"kfcollect {__version__}"}, "timestamp": datetime.utcnow().isoformat() + "Z"}
        if i == 0 and description:
            embed["description"] = description
        if page:
            embed["fields"] = page
        ok &= _post(cfg, embed)
    return ok


def summary_fields(s: dict) -> list:
    pk = s["polls_hour"]
    def rate(kind):
        k = pk.get(kind)
        return f"{k['ok']}/{k['n']} ok · {k['ms']} ms" if k else "—"
    focus = "\n".join(
        f"**{v['name']}**: {v['vehicles']} bus{'es' if v['vehicles'] != 1 else ''}"
        + (f", avg {v['avg_dev_s']:+d}s" if v["avg_dev_s"] is not None and v["vehicles"] else "")
        for v in s["focus"].values())
    return [
        ("Collector", f"heartbeat {_fmt_age(s['heartbeat_age_s'])}\nlast good poll {_fmt_age(s['last_ok_age_s'])}", True),
        ("Last hour", f"routes: {rate('tfr')}\narrivals: {rate('arrivals')}\nGTFS-RT: {rate('gtfs_rt')}", True),
        ("Focus routes (now)", focus or "—", False),
        ("Rows", f"{s['rows_hour']:,} last hour · {s['rows_today']:,} today · {s['rows_total']:,} total\n"
                 f"{s['arrivals_today']:,} arrival predictions · {s['schedules_today']:,} trip schedules today", False),
        ("Derived today", f"{s['passages_today'] if s['passages_today'] is not None else '—'} stop passages · "
                          f"{s['gap_minutes_today'] or 0} gap minutes", True),
        ("Storage", f"DB {s['db_gb']:.2f} GB · raw today {s['raw_today_mb']:.0f} MB · raw total "
                    f"{s['raw_total_gb']:.2f} GB · free {s['free_gb']:.1f} GB" if s["free_gb"] is not None else "—", True),
    ]


# --- checks ------------------------------------------------------------------------------------

def _load_state(cfg: Config) -> dict:
    p = cfg.state_dir / "status.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"alerts": {}, "last_event_id": 0, "last_summary": 0, "last_daily": ""}


def _save_state(cfg: Config, state: dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "status.json").write_text(json.dumps(state))


def check(db: sqlite3.Connection, cfg: Config, force_summary: bool = False) -> dict:
    s = gather(db, cfg)
    state = _load_state(cfg)
    if not state.get("last_event_id"):
        # First run: start from the newest event instead of replaying history.
        state["last_event_id"] = _scalar(db, "SELECT MAX(id) FROM events") or 0
    now = time.time()
    conditions = {
        "collector_down": (s["heartbeat_age_s"] is None or s["heartbeat_age_s"] > 180,
                           f"Last heartbeat {_fmt_age(s['heartbeat_age_s'])}. Is kfcollect.service running? "
                           "Check: sudo systemctl status kfcollect"),
        "feed_failing": ((s["heartbeat"].get("consecutive_failures") or 0) >= 6,
                         f"{s['heartbeat'].get('consecutive_failures')} poll cycles in a row failed; "
                         f"last good poll {_fmt_age(s['last_ok_age_s'])}."),
        "no_vehicles": (s["in_service"] and sum(v["vehicles"] for v in s["focus"].values()) == 0
                        and (s["heartbeat_age_s"] or 999) < 180,
                        "No tracked buses on the focus routes for 10+ minutes during weekday service hours."),
        "disk_low": (s["free_gb"] is not None and s["free_gb"] < cfg.min_free_gb * 2,
                     f"Only {s['free_gb']:.1f} GB free on the data disk." if s["free_gb"] is not None else ""),
    }
    for key, (active, msg) in conditions.items():
        a = state["alerts"].get(key)
        if active and not a:
            send(cfg, f"🚨 {key.replace('_', ' ')}", msg, RED)
            state["alerts"][key] = {"since": now, "notified": now}
        elif active and now - a["notified"] > REMIND_EVERY_S:
            send(cfg, f"⏰ still: {key.replace('_', ' ')}",
                 f"{msg}\nOngoing for {(now - a['since']) / 3600:.1f} h.", YELLOW)
            a["notified"] = now
        elif not active and a:
            send(cfg, f"✅ resolved: {key.replace('_', ' ')}",
                 f"Lasted {(now - a['since']) / 60:.0f} min.", GREEN)
            del state["alerts"][key]

    # Forward collector events (new alerts, schedule versions, disk pruning, errors).
    events = db.execute("SELECT id, ts, level, kind, message FROM events WHERE id > ? ORDER BY id LIMIT 20",
                        (state.get("last_event_id", 0),)).fetchall()
    for e in events:
        state["last_event_id"] = e["id"]
        if e["kind"] in ("start", "stop") or e["level"] == "debug":
            continue
        color = {"error": RED, "warning": YELLOW}.get(e["level"], BLUE)
        send(cfg, f"{e['kind'].replace('_', ' ')}", e["message"], color)

    # Periodic summary.
    if force_summary or now - state.get("last_summary", 0) >= cfg.summary_every_hours * 3600 - 60:
        ok = not state["alerts"]
        send(cfg, "📊 Collection status" + ("" if ok else " (issues open)"),
             f"Collecting since {_fmt_since(s['first_data'], cfg)}", GREEN if ok else YELLOW, summary_fields(s))
        state["last_summary"] = now

    _save_state(cfg, state)
    return s


def _fmt_since(ms, cfg: Config) -> str:
    if not ms:
        return "—"
    dt = datetime.fromtimestamp(ms / 1000, cfg.timezone)
    days = (datetime.now(cfg.timezone) - dt).days
    return f"{dt:%a %b %d %H:%M} ({days} day{'s' if days != 1 else ''})"

