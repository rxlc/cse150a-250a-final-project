# kfcollect — campus shuttle data collector

Always-on logger for the **Campus Bus Reliability (HMM)** project and the submitted proposal
**"Real-Time Probabilistic Prediction of Triton Transit Shuttle Delays"**. It stores a superset of
what both need, derives the observation tables, and reports health to Discord. The models themselves
live in the project repository, not here.

## Install (Ubuntu 22.04, Azure VM)

```bash
# from your computer
scp kfcollect.zip azureuser@<server>:~
# on the server
unzip kfcollect.zip && cd kfcollect
sudo ./install.sh --webhook "https://discord.com/api/webhooks/…"
kfcollect status --test          # a test message should appear in Discord
```

The installer creates a `kfcollect` system user, puts code in `/opt/kfcollect` (with a venv), config in `/etc/kfcollect/config.env`, data in `/var/lib/kfcollect`, and starts:

| Unit | When | Does |
| --- | --- | --- |
| `kfcollect.service` | always (restarts on failure) | the collector |
| `kfcollect-status.timer` | every 5 min | alerts + summary every 6 h |
| `kfcollect-derive.timer` | hourly | stop passages, segments, 5-min intervals, vehicle steps |
| `kfcollect-report.timer` | daily 22:15 PT | daily report with derived counts |

`sudo ./uninstall.sh` removes it (add `--purge` to delete data too).

## Updating

From your computer, in the folder with `kfcollect.zip` and `deploy.sh`:

```bash
./deploy.sh azureuser@<server>                                 # copy + update in one step
SSH_OPTS="-i ~/.ssh/azure.pem" ./deploy.sh azureuser@<server>   # with a key file
```

On Windows (PowerShell), with `kfcollect.zip` and `deploy.ps1` in the same folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy.ps1 -Server azureuser@<server>     # add -Key C:\path\key.pem if needed
```

Or by hand on the server after `scp kfcollect.zip azureuser@<server>:~`:

```bash
rm -rf ~/kfcollect && python3 -m zipfile -e ~/kfcollect.zip ~ && chmod +x ~/kfcollect/*.sh
sudo ~/kfcollect/update.sh            # --no-report to skip the fresh report
```

`update.sh` keeps the previous version, copies the new code, adds any **new** config settings with their
defaults (existing values untouched), restarts the collector and checks that it stays up, **rolling back
automatically** if it doesn't. It then runs the self test, posts "Updated X → Y" to Discord, re-derives the
last 3 days and posts a fresh report. Manual rollback: `sudo ~/kfcollect/update.sh --rollback`.
Data collection pauses only for the restart (a few seconds).

## Changelog

**1.3.0**
* The interval HMM's observation is **excess wait**: how much longer than the timetable a rider waits for the
  next bus, whichever trip or bus it is (see `intervals_5min` below). Trip delay is kept as `delay_level`.
* Loops: a bus driving up to the start no longer matches the *end* of its first trip (which lost that whole
  trip, for every bus, every day). About 1,000 more UCSD passages over 12 days.
* `derive` and `report` run one at a time (they wait for each other), and `update.sh` stops derivations still
  running old code before re-deriving; if it stops an earlier update's re-derive, it re-derives every day.

**1.2.0** (found by checking the first 12 days of real data)
* **Route attribution fixed.** `trips-for-route` also lists other routes' trips in shared blocks (Inside/Outside
  Loop share buses; so do SuperLoop 201/202). Derived tables used the route a row was *polled under*, so most
  Outside Loop trips were filed as Inside Loop (Outside Loop had 92 stop passages in 12 days) and 201/202 were
  swapped: 7.9% of passages were mislabelled. Now each trip's own `routeId` is used. On real data: 100% correct,
  Outside Loop 21 → 1,571 passages over 4 test days.
* **Daylight saving.** Triton Transit's GTFS times are wall-clock from midnight, but OneBusAway counts them
  from "noon minus 12 h", so on Nov 1 every UCSD scheduled time was an hour late: OneBusAway's deviation and
  derived delays read about −3,600 s all day. UCSD delays (and OneBusAway's deviation in `stop_passages`) are
  now corrected (`KF_WALLCLOCK_AGENCIES`). MTS was not affected. Nov 1 UCSD median delay: −3,541 s → +47 s.
* **Gaps** are measured from the time between successful polls. 1.1.0 counted every other minute overnight
  as a gap (929 false gap minutes in 12 days; the real number was 0).
* **Status** counts each bus under its trip's own route over the last 2 minutes (1.1.0's fix didn't work:
  the same bus never appears on two routes in one poll; it was the 10-minute window plus the attribution bug).
* `update.sh` re-derives **every day** in a background systemd job when the derivation changed, then posts a
  report; `export` now includes `shapes`.

**1.1.0**
* Regime and interval HMMs are fit **per route and direction** (as the proposal specifies). 1.0.0 pooled the
  three loops, so its "regimes" mostly captured route differences (durations of thousands of segments).
  Reports now show, per route: each regime's delay gained, share of time and typical duration in minutes.
* New **delay check** in reports: GPS-derived stop delays vs OneBusAway's own `scheduleDeviation` per route
  (median difference, MAE, correlation; ⚠️ when they disagree).
* Status counts each bus under the trip it is running: an interlined Inside/Outside Loop bus was counted on both.
* Overnight: empty responses aren't archived; polling slows to every 2 min when no trips are listed; arrival
  predictions only while buses run on campus routes. Measured at night: 0 raw records instead of 119 per 200 s.
* Raw archive written as complete gzip members every 60 s: files stay valid after a crash (`kill -9` tested).
  Shutdown takes under a second.
* Long Discord reports are split across messages instead of being rejected.
* `update.sh` / `deploy.sh`, `kfcollect status --note`.

## Commands

```bash
kfcollect status --print         # health and counts, printed here
kfcollect status --summary       # post a summary to Discord now
kfcollect derive [--days N | --full]
kfcollect report                 # what the daily timer runs
kfcollect selftest               # endpoints and database
kfcollect status --note "text"   # post a note to Discord
kfcollect export [--since 2026-10-27]   # CSVs in /var/lib/kfcollect/exports/*.zip
journalctl -u kfcollect -f       # collector logs
```

Copy data to your computer: `scp azureuser@<server>:/var/lib/kfcollect/exports/kf-export-*.zip .`
(or the whole SQLite file after `sudo sqlite3 /var/lib/kfcollect/kf.sqlite ".backup /tmp/kf.sqlite"`).

## What is collected

| Source | How often | Stored as |
| --- | --- | --- |
| OneBusAway `trips-for-route` (status **and** schedule) for 14 routes | every 30 s; every 2 min when no route lists any trip (overnight) | `trip_status` (every field in the proposal's table, one row per trip per poll; unknown fields kept in `extra`) |
| …each trip's schedule | once per trip per service day | `trip_schedules`, `stop_times` |
| …reference tables | once per day | `ref_trips`, `ref_stops`, `ref_routes` |
| …service alerts | when a `situationId` appears (last_seen updated) | `situations` |
| …route shapes | once | `shapes` |
| `arrivals-and-departures-for-stop`, 37 focus-route stops | one pass every 2 min while buses run on routes serving those stops | `arrivals` (the predictions riders saw: a calibration baseline) |
| `schedule-for-stop`, same stops | daily | `timetables` (the published timetable per stop per day) |
| Triton Transit **GTFS static** (`gtfs.zip`) | daily, kept when it changes | `data/gtfs/*.zip`, `gtfs_versions` |
| Triton Transit **GTFS-Realtime** (vehicle positions, trip updates, alerts) | every 30 s (2 min overnight) | raw archive (protobuf) |
| **Every response above that has content** | — | `data/raw/YYYY-MM-DD/HH.jsonl.gz` |
| Collector health | every poll / minute | `polls`, `heartbeat`, `events`, `heartbeat.json` |

Routes: focus = Inside Loop, Outside Loop, Mesa Loop; also SIO, Grocery-Convoy, Regents, MTS SuperLoop
201/202/204, MTS 30/41/237, Blue Line trolley, NCTD 101 (they share campus stops). Edit `KF_FOCUS_ROUTES`
/ `KF_EXTRA_ROUTES` in the config.

### How this covers both proposals

| | HMM proposal (PDF) | Submitted proposal | Here |
| --- | --- | --- | --- |
| Poll rate | 30 s | 30 s | 30 s |
| Schedule source | `trips-for-route` schedule | published Transportation Services timetables | both, plus the official GTFS static feed and daily `schedule-for-stop` |
| Raw responses | gzipped, 2 days | stored | gzipped, kept (pruned only when disk runs low) |
| Derived | per-segment delay gained; vehicle traces | stop passages → 5-min intervals → deviation levels | `segments`, `vehicle_steps`, `stop_passages`, `intervals_5min` (levels from rider excess wait, trip delay kept alongside) |
| Ground truth for disruptions | alerts | — | `situations` + GTFS-RT alerts |

### Not wasting space overnight

* Empty answers (a route with no trips and no alerts, a GTFS-RT feed with no entities) are **not** written
  to the raw archive; the `polls` table still records each request, so coverage is provable.
* After 2 minutes with no trips listed on any route, polling slows to every 2 minutes
  (`KF_IDLE_POLL_INTERVAL_S`) and returns to 30 s as soon as a trip appears. Worst case, the first minute or
  two of the first morning trip is polled less often.
* Arrival predictions are polled only while buses run on routes that serve the campus stops.

Measured at 2:45 a.m. over 200 s: 56 route polls instead of 98, and zero raw records instead of 119.

### Raw archive format

Records are buffered and appended as a complete gzip member every 60 s (and on shutdown), so each
`HH.jsonl.gz` is always a valid gzip file (`zcat`, `gzip.open`), even after a crash or `kill -9`. A crash
loses at most the last minute of raw records; the parsed tables are written immediately.

```bash
zcat /var/lib/kfcollect/raw/2026-10-28/08.jsonl.gz | head -1 | python3 -m json.tool | head
```

## Derived tables (`kfcollect derive`)

* **`stop_passages`**: when each trip passed each scheduled stop and its delay (actual − scheduled).
  Progress is measured from the bus's **own GPS fix** (`lastKnownLocation` at `lastLocationUpdateTime`)
  projected onto the trip shape, not OBA's `distanceAlongTrip`: on this feed that was observed equal to
  `scheduledDistanceAlongTrip`, which would make every derived delay zero. OBA's `scheduleDeviation` at
  the nearest poll is kept alongside (`obs_deviation_s`) for comparison. Departure from the first stop
  is when the bus is 25 m past it; arrival at the last stop is 20 m before it. On a synthetic bus driven
  along the real Inside Loop shape with 30 s fixes, intermediate stops were recovered within ~4 s
  (mean 1.8 s), and the first and last stops within ~12 s. A trip's first fix can't be matched farther
  along than the timetable allows (5 min early at most): on loops, a bus driving up to the start used to
  match the *end* of the trip, and the whole real trip was lost behind it (the first trip of every bus,
  every day, before 1.3.0).
* **`segments`**: delay gained between consecutive stops, binned (≤0, 0–30, 30–90, 90–180, >180 s).
* **`intervals_5min`**: per route/direction 5-minute slots. The observation (`level`) is **excess wait**:
  for a rider who shows up at a random moment in the slot, how much longer they wait for the next bus
  (any trip, any bus) than the timetable promises, averaged over the route's stops (`sched_wait_s`,
  `wait_s`, `excess_wait_s`, worst stop in `max_excess_s`). Levels: 0 shorter / 1 normal / 2 longer /
  3 much longer (boundaries −60, 120, 300 s; `KF_WAIT_LEVELS`). A bus running a whole headway behind is
  then *not* late once the gaps between buses are back to normal, but bunching and missing buses are.
  Each trip's own delay is kept alongside (`mean_delay` … and `delay_level`, early / on time / late / very
  late). Details: stops a tracked trip passed between two observed stops get its delay interpolated (and up
  to 2 stops past its first/last observed one); a trip's last stop isn't a boarding chance and is left
  out; riders are counted only within each stop's service span for the day, up to that day's last tracked
  bus there, and not within 30 min after a collector gap or before the newest poll (the next run fills
  that in). `missed_trips` counts scheduled trips starting in the slot that no bus was tracked on.
* **`vehicle_steps`**: per-bus steps with speed bin (stopped/slow/moving), at-stop and layover-zone flags (M1).
* **`trip_events`** (bus swaps) and **`gaps`** (minutes with no successful poll).

Preprocessing from the proposal is applied: dedupe on `lastLocationUpdateTime`, drop untracked rows
(`predicted = false`, blank `phase`, no vehicle), drop fixes > 2 min stale, and use only rows where
`activeTripId` is the trip itself (other rows of a block describe the bus on its current trip).
Raw data is never modified, so `kfcollect derive --full` can rebuild everything after changing the method.

## Discord messages

* 🚨 **alerts** (and ✅ when resolved, ⏰ reminders every 2 h): collector down (no heartbeat 3 min),
  feed failing (6 cycles), no buses on focus routes for 10 min during weekday service hours, low disk.
* Collector events: new service alerts, new GTFS versions, disk pruning, errors.
* 📊 summary every 6 h; 📅 daily report at 22:15 PT with derived counts.

## Things to know about the data (first 12 days)

* OneBusAway's `distanceAlongTrip` always equals `scheduledDistanceAlongTrip` and `lastKnownDistanceAlongTrip`
  is always 0 on this feed: progress must come from GPS (as derived here). Occupancy fields are never filled.
* **Nov 1 (DST)**: corrected in 1.2.0, but UCSD routes had about a third of a normal Sunday's passages that day;
  consider excluding it or treating it with care.
* **Oct 28, 16:45–19:37**: one Outside Loop bus (`…fc1470`) ran 30–35 min behind its trips while moving
  normally: a full lap (two headways) behind, so riders saw normal gaps. By trip delay that's 3 hours "very late";
  by rider wait (1.3.0) only 16:10–16:35 is bad, while the two buses were bunched. An interval HMM fitted on
  9 weekdays puts 16:45–19:40 in its worst state on trip delay, and only 16:10–16:30 on excess wait.
* **Mesa Loop "runs early"** (58% of slots by trip delay) is the same kind of artifact: by rider wait it's 92%
  normal. Riders who time their arrival to the timetable on long-headway routes do care about early buses;
  trip delay (`delay_level`) is still there for that.
* **Nov 2, 11:58–14:17**: Outside Loop bus `…b526b` was missing; 4 trips never ran (`missed_trips`) and
  riders waited about twice as long. A real event the wait levels catch.
* Wait levels by route (Oct 28–Nov 9, weekdays + Inside Loop weekends): normal 84% (Outside), 94% (Inside),
  92% (Mesa); much longer 0–2.3%.
* **Weekends**: only Inside Loop runs (about 10:15–18:05). Its weekday and weekend behavior differ enough that a
  single HMM spends a state on day type; fit them separately or add day type as a covariate.
* Our derived delays track OneBusAway's deviation closely (correlation about 0.9) with a steady +10–20 s offset:
  ours is the moment the bus reaches the stop; OneBusAway's is its estimate at the nearest poll.

## Things to know

* **No service** Nov 11 (Veterans Day), Nov 26–27 (Thanksgiving), per the GTFS calendar; "no buses"
  alerts skip those dates. Fall service ends Dec 12.
* **DST ends Nov 1**: scheduled times are `serviceDate + seconds`; check one stop passage that day against
  the printed schedule, as the proposal says.
* **API**: the default is the keyless Wayfinder proxy, which is undocumented. If you get a OneBusAway key,
  switch `KF_OBA_FLAVOR=standard` (see the config).
* **Disk**: expect a few hundred MB/day of raw archive during service; `KF_MIN_FREE_GB` triggers pruning of
  the oldest raw days (never the last 3, never the SQLite tables). `kfcollect status --print` shows usage.
* **Clock**: rows use the server's `currentTime`; NTP should be on (`timedatectl`).
