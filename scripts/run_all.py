"""
Run every step in order and stop at the first one that fails.

Steps 00 to 07 verify the physics, the sensor model and the simulator, and
build the comparison baselines. Steps 08 to 13 fetch real weather, identify
the room from logged data, design the planning controller and evaluate it
across five climates.

The scripts in scripts/legacy belong to the first version of the project and
are no longer part of the pipeline; see scripts/legacy/README.md.

The steps depend on each other: the physics feeds the data collection, the
data feeds the model, the model feeds the controller. So if any constant in
config.py changes, this is the way to rebuild everything consistently rather
than trying to remember which steps are downstream of what.
"""

import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
STEPS = [
    ("00", "core invariants", "../tests/test_core.py"),
    ("01", "room model", "step01_room.py"),
    ("02", "verify the thermal model", "step02_validate.py"),
    ("03", "synthetic weather generator", "step03_weather.py"),
    ("04", "window and fan effects", "step04_window_fan.py"),
    ("05", "sensor model", "step05_sensors.py"),
    ("06", "simulator and logger", "step06_sim_run.py"),
    ("07", "timer and threshold-rule baselines", "step07_baselines.py"),
    ("08", "fetch real weather", "fetch_weather.py"),
    ("09", "collect identification data", "step15_collect_real.py"),
    ("10", "identify the room dynamics", "step16_identify.py"),
    ("11", "controller design and ablations", "step17_controller.py"),
    ("12", "evaluation by climate", "step18_evaluate.py"),
    ("13", "figures and write-up", "step19_report.py"),
]

def main(argv):
    only = set(argv[1:])
    results = []
    for num, title, script in STEPS:
        if only and num not in only:
            continue
        path = HERE / script
        if not path.exists():
            print(f"\n=== Step {num}: {title} -- SKIPPED (no {script} yet)")
            continue
        print(f"\n{'='*78}\n=== Step {num}: {title}\n{'='*78}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(path)], cwd=HERE.parent)
        secs = time.time() - t0
        results.append((num, title, proc.returncode == 0, secs))
        if proc.returncode != 0:
            print(f"\n!!! Step {num} FAILED after {secs:.1f}s. Stopping.")
            break

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for num, title, ok, secs in results:
        print(f"  Step {num}  {'PASS' if ok else 'FAIL'}  {secs:6.1f}s  {title}")
    total = sum(r[3] for r in results)
    n_ok = sum(1 for r in results if r[2])
    print(f"\n  {n_ok}/{len(results)} steps passed in {total:.1f}s")
    # Record it so the report can quote the real figure instead of a guess.
    if not only and all(r[2] for r in results):
        out = HERE.parent / "results" / "runtime.json"
        out.write_text(json.dumps({"total_seconds": round(total, 1),
                                   "steps": len(results)}, indent=2))
    return 0 if all(r[2] for r in results) else 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
