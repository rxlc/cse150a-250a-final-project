"""Configuration from /etc/kfcollect/config.env (KEY=VALUE), overridden by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_CONFIG_FILE = "/etc/kfcollect/config.env"

# Inside Loop, Outside Loop, Mesa Loop.
DEFAULT_FOCUS = "UCSD_1040,UCSD_1030,UCSD_1050"
# Other student-heavy routes sharing campus stops: SIO, Grocery-Convoy, Regents Express,
# MTS SuperLoop (201/202/204), MTS 30, 41, 237, Blue Line trolley, NCTD 101.
DEFAULT_EXTRA = "UCSD_1020,UCSD_1060,UCSD_1010,MTS_201,MTS_202,MTS_204,MTS_30,MTS_41,MTS_237,MTS_510,NCTD_301"


def _read_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Config:
    data_dir: Path
    oba_base_url: str
    oba_flavor: str  # "wayfinder" (keyless proxy) or "standard" (/api/where with a key)
    oba_api_key: str
    focus_routes: list[str]
    extra_routes: list[str]
    poll_interval_s: int
    idle_poll_interval_s: int
    arrivals_interval_s: int
    gtfs_base: str
    gtfs_rt_interval_s: int
    timezone: ZoneInfo
    discord_webhook_url: str
    summary_every_hours: int
    raw_keep_days: int
    min_free_gb: float
    service_start: str
    service_end: str
    wallclock_agencies: list[str] = field(default_factory=list)
    deviation_levels: list[int] = field(default_factory=list)
    wait_levels: list[int] = field(default_factory=list)
    gained_bins: list[int] = field(default_factory=list)

    @property
    def routes(self) -> list[str]:
        seen: list[str] = []
        for r in self.focus_routes + self.extra_routes:
            if r not in seen:
                seen.append(r)
        return seen

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kf.sqlite"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def gtfs_dir(self) -> Path:
        return self.data_dir / "gtfs"

    @property
    def heartbeat_path(self) -> Path:
        return self.data_dir / "heartbeat.json"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get("KF_CONFIG", DEFAULT_CONFIG_FILE)
    values = _read_env_file(path)
    values.update({k: v for k, v in os.environ.items() if k.startswith("KF_")})

    def get(key: str, default: str) -> str:
        return values.get(key, default) or default

    cfg = Config(
        data_dir=Path(get("KF_DATA_DIR", "/var/lib/kfcollect")),
        oba_base_url=get("KF_OBA_BASE_URL", "https://wayfinder.ucsd.onebusawaycloud.com/api/oba/"),
        oba_flavor=get("KF_OBA_FLAVOR", "wayfinder"),
        oba_api_key=values.get("KF_OBA_API_KEY", ""),
        focus_routes=_csv(get("KF_FOCUS_ROUTES", DEFAULT_FOCUS)),
        extra_routes=_csv(values.get("KF_EXTRA_ROUTES", DEFAULT_EXTRA)),
        poll_interval_s=int(get("KF_POLL_INTERVAL_S", "30")),
        idle_poll_interval_s=int(get("KF_IDLE_POLL_INTERVAL_S", "120")),
        arrivals_interval_s=int(get("KF_ARRIVALS_INTERVAL_S", "120")),
        gtfs_base=get(
            "KF_GTFS_BASE",
            "https://api.us.sparelabs.com/v1/fixedRoute/public/21c87bba-e136-41bc-ba9a-191bdb3e08e4",
        ),
        gtfs_rt_interval_s=int(get("KF_GTFS_RT_INTERVAL_S", "30")),
        timezone=ZoneInfo(get("KF_TIMEZONE", "America/Los_Angeles")),
        discord_webhook_url=values.get("KF_DISCORD_WEBHOOK_URL", ""),
        summary_every_hours=int(get("KF_SUMMARY_EVERY_HOURS", "6")),
        raw_keep_days=int(get("KF_RAW_KEEP_DAYS", "0")),
        min_free_gb=float(get("KF_MIN_FREE_GB", "2")),
        service_start=get("KF_SERVICE_START", "07:00"),
        service_end=get("KF_SERVICE_END", "22:00"),
        # Agencies whose GTFS times are wall-clock from midnight (matters on DST days only).
        wallclock_agencies=_csv(get("KF_WALLCLOCK_AGENCIES", "UCSD")),
        # Deviation level boundaries (s): early < -60 <= on time <= 120 < late <= 300 < very late.
        deviation_levels=[int(x) for x in _csv(get("KF_DEVIATION_LEVELS", "-60,120,300"))],
        # Excess wait level boundaries (s), the interval HMM's observation:
        # shorter < -60 <= normal <= 120 < longer <= 300 < much longer.
        wait_levels=[int(x) for x in _csv(get("KF_WAIT_LEVELS", "-60,120,300"))],
        # Delay-gained-per-segment bins (s): <=0, 0-30, 30-90, 90-180, >180 (per the HMM proposal).
        gained_bins=[int(x) for x in _csv(get("KF_GAINED_BINS", "0,30,90,180"))],
    )
    return cfg
