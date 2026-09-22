###############################################################################
# Copyright (c) 2025 Giri M. Kumar, Andrea L'Afflitto. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
###############################################################################
#
###############################################################################
# File:        wrapper.py
# Authors:     Giri M. Kumar
# Date:        September 20, 2026
# For info:    Andrea L'Afflitto
#              a.lafflitto@vt.edu
#
# Description:
#     Orchestrator for the simulator's "wrapper" mode: runs build/acsl_sim
#     repeatedly, "total_runs" times (config/wrapper-config.yaml), keeping up
#     to "concurrent_runs" simulations alive at once. Runs are launched in a
#     rolling fashion -- new ones are started (staggered 1 second apart) as
#     soon as a slot frees up -- until the full total has been executed.
#
#     A run "erroring out" is detected the same way for every run: this
#     script polls the child process and checks its exit code once it exits.
#     A non-zero exit is logged as a failure and the freed slot is simply
#     handed to the next run; nothing about a failing run blocks the rest of
#     the session.
#
#     Requires config/sim-config.yaml's mode.enable_wrapper to be true (this
#     only controls how build/acsl_sim lays out its own log directory; this
#     script still refuses to run anything if it's false, since wrapper-style
#     concurrent runs would otherwise collide in a single-run log directory).
#
#     Run from anywhere with:
#         python3 scripts/wrapper.py
###############################################################################

#!/usr/bin/env python3

from __future__ import annotations

import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional

try:
    import yaml
except ImportError:
    print(
        "PyYAML is required but not installed.\n\n"
        "Install it with:\n\n"
        "    pip3 install pyyaml\n",
        file=sys.stderr,
    )
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
CONFIG_DIR = REPO_DIR / "config"
BUILD_DIR = REPO_DIR / "build"
EXECUTABLE = BUILD_DIR / "acsl_sim"
SIM_CONFIG_PATH = CONFIG_DIR / "sim-config.yaml"
WRAPPER_CONFIG_PATH = CONFIG_DIR / "wrapper-config.yaml"

# Runs are always at least this far apart, both for the initial concurrent
# batch and for every rolling replacement. This isn't just pacing -- two runs
# launched in the same instant would both scan the wrapper's run_N
# directories (sim-logger.cpp, ConfigureLogDirectoryWrapperRun STEP 4) before
# either one exists on disk, and could pick the same run number.
STAGGER_SECONDS = 1.0
POLL_INTERVAL_SECONDS = 1.0

# ANSI color codes -- mirrors include/sim-utilities/sim-messages.hpp so
# wrapper.py's terminal output reads consistently with the simulator's own.
COLOR_BLUE = "\033[1;34m"
COLOR_GREEN = "\033[1;32m"
COLOR_RED = "\033[1;31m"
COLOR_ORANGE = "\033[38;5;214m"
COLOR_RESET = "\033[0m"
# "\r" alone (no "\x1b[2K" erase-line code -- the GUI's console only parses
# SGR "\x1b[...m" color codes, so "2K" showed up as literal text there).
# Its other job -- the one that actually matters -- is that _status_line
# below is the only print() with flush=True: stdout is block-buffered, not
# line-buffered, when it's piped to the GUI's subprocess rather than a real
# terminal, so without that periodic flush every _info/_warning line would
# just sit in the buffer, unseen, until the process exits.
CLEAR_LINE = "\r"


def _info(message: str) -> None:
    print(f"{CLEAR_LINE}{COLOR_GREEN}[INFO] [--ACSL WRAPPER--] {message}{COLOR_RESET}")


def _warning(message: str) -> None:
    print(f"{CLEAR_LINE}{COLOR_ORANGE}[WARN] [--ACSL WRAPPER--] {message}{COLOR_RESET}")


def _error(message: str) -> None:
    print(f"{CLEAR_LINE}{COLOR_RED}[ERROR] [--ACSL WRAPPER--] {message}{COLOR_RESET}", file=sys.stderr)


def _status_line(completed: int, total: int, running: int, succeeded: int, failed: int, elapsed: float) -> None:
    print(
        f"{CLEAR_LINE}{COLOR_BLUE}[WRAPPER] runs {completed}/{total} complete | "
        f"{running} running | {succeeded} ok | {failed} failed | elapsed {elapsed:0.0f}s{COLOR_RESET}",
        end="",
        flush=True,
    )


# Reads a YAML config file, exiting with an error if it can't be opened --
# mirrors the ifstream check in simlog::ConfigureLogDirectoryWrapperRun.
def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        _error(f"COULD NOT OPEN CONFIG FILE: {path}")
        sys.exit(1)
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


# Mirrors platforms::validateExclusiveSelection: exactly one platform must be
# selected in sim-config.yaml, and wrapper.py needs to know which one to find
# the right sim-log/<platform>/wrapper directory.
def _get_active_platform(sim_config: dict) -> str:
    platforms = sim_config.get("platform", {}) or {}
    active = [name for name, enabled in platforms.items() if enabled]
    if len(active) != 1:
        _error(f"EXPECTED EXACTLY ONE ACTIVE PLATFORM IN sim-config.yaml, FOUND: {active}")
        sys.exit(1)
    return active[0]


@dataclass
class WrapperSettings:
    total_runs: int
    concurrent_runs: int


def _read_wrapper_settings(wrapper_config: dict) -> WrapperSettings:
    settings = wrapper_config.get("wrapper", {}) or {}
    total_runs = int(settings.get("total_runs", 0))
    concurrent_runs = int(settings.get("concurrent_runs", 1))

    if total_runs <= 0:
        _error("wrapper-config.yaml: 'total_runs' must be a positive integer")
        sys.exit(1)
    if concurrent_runs <= 0:
        _warning("wrapper-config.yaml: 'concurrent_runs' must be positive, defaulting to 1")
        concurrent_runs = 1
    if concurrent_runs > total_runs:
        _warning(f"'concurrent_runs' ({concurrent_runs}) exceeds 'total_runs' ({total_runs}), clamping")
        concurrent_runs = total_runs

    return WrapperSettings(total_runs=total_runs, concurrent_runs=concurrent_runs)


# Path to the session marker the C++ side reads to survive a midnight
# rollover mid-session (sim-logger.cpp, ConfigureLogDirectoryWrapperRun
# STEP 3.1 -- currently commented out/experimental there). wrapper.py owns
# closing it out: the executable can only ever see "am I still active", it
# has no way to know when the *whole* wrapper session is done.
def _session_marker_path(platform: str) -> Path:
    return REPO_DIR / "sim-log" / platform / "wrapper" / ".session.bak"


def _deactivate_session_marker(platform: str) -> None:
    marker = _session_marker_path(platform)
    if not marker.parent.is_dir():
        return  # No run ever got far enough to create the wrapper directory.
    try:
        marker.write_text("inactive\n")
        _info(f"WRAPPER SESSION MARKED INACTIVE: {marker}")
    except OSError as error:
        _warning(f"COULD NOT WRITE SESSION MARKER INACTIVE: {marker} ({error})")


@dataclass
class _RunHandle:
    run_index: int
    process: subprocess.Popen
    log_file: BinaryIO
    started_at: float


# Launches one simulation. Each run's console output is captured to its own
# log file (rather than inherited) so concurrent runs' output doesn't
# interleave on top of the status line below.
def _launch_run(run_index: int, console_dir: Path) -> _RunHandle:
    log_path = console_dir / f"run_{run_index + 1:03d}.log"
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        ["./acsl_sim"],
        cwd=str(BUILD_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _info(f"STARTED RUN {run_index + 1} (PID {process.pid}) -> {log_path.relative_to(REPO_DIR)}")
    return _RunHandle(run_index=run_index, process=process, log_file=log_file, started_at=time.monotonic())


# Stops any still-running simulations (SIGINT first, then SIGKILL if a run
# doesn't wind down in time) and closes their log files. Runs on every exit
# path -- normal completion, Ctrl+C, or an unexpected error.
def _terminate_all(active: Dict[int, _RunHandle]) -> None:
    for handle in active.values():
        if handle.process.poll() is None:
            handle.process.send_signal(signal.SIGINT)

    deadline = time.monotonic() + 5.0
    for handle in active.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            handle.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait()
        handle.log_file.close()


def main() -> int:
    sim_config = _load_yaml(SIM_CONFIG_PATH)
    if not sim_config.get("mode", {}).get("enable_wrapper", False):
        _warning("mode.enable_wrapper is false in sim-config.yaml -- nothing to do")
        return 0

    platform = _get_active_platform(sim_config)
    settings = _read_wrapper_settings(_load_yaml(WRAPPER_CONFIG_PATH))

    if not EXECUTABLE.is_file():
        _error(f"SIMULATOR EXECUTABLE NOT FOUND: {EXECUTABLE} (build it first)")
        return 1

    console_dir = REPO_DIR / "sim-log" / platform / "wrapper" / "console"
    console_dir.mkdir(parents=True, exist_ok=True)

    _info(
        f"STARTING WRAPPER SESSION: {settings.total_runs} total run(s), "
        f"{settings.concurrent_runs} concurrent, platform '{platform}'"
    )

    active: Dict[int, _RunHandle] = {}
    next_run_index = 0
    completed = 0
    succeeded = 0
    failed = 0
    last_launch_time: Optional[float] = None
    session_start = time.monotonic()

    # Starts the next queued run, if any are left, staggered STAGGER_SECONDS
    # after the previous launch. Called both to prime the initial concurrent
    # batch and to backfill a slot freed up by a finished run.
    def launch_next() -> None:
        nonlocal next_run_index, last_launch_time
        if next_run_index >= settings.total_runs:
            return
        if last_launch_time is not None:
            wait_for = STAGGER_SECONDS - (time.monotonic() - last_launch_time)
            if wait_for > 0:
                time.sleep(wait_for)
        active[next_run_index] = _launch_run(next_run_index, console_dir)
        next_run_index += 1
        last_launch_time = time.monotonic()

    try:
        for _ in range(min(settings.concurrent_runs, settings.total_runs)):
            launch_next()

        while completed < settings.total_runs:
            for run_index, handle in list(active.items()):
                returncode = handle.process.poll()
                if returncode is None:
                    continue  # Still running.

                del active[run_index]
                handle.log_file.close()
                completed += 1
                duration = time.monotonic() - handle.started_at
                if returncode == 0:
                    succeeded += 1
                    _info(f"RUN {run_index + 1} FINISHED OK in {duration:0.1f}s")
                else:
                    failed += 1
                    _warning(f"RUN {run_index + 1} EXITED WITH ERROR (code {returncode}) after {duration:0.1f}s -- moving on")
                launch_next()

            elapsed = time.monotonic() - session_start
            _status_line(completed, settings.total_runs, len(active), succeeded, failed, elapsed)
            time.sleep(POLL_INTERVAL_SECONDS)

        print()  # Move past the in-place status line before the final summary.
        _info(f"WRAPPER SESSION COMPLETE: {succeeded} succeeded, {failed} failed, out of {settings.total_runs} total run(s)")
        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        print()
        _warning("WRAPPER INTERRUPTED -- STOPPING ACTIVE RUNS")
        return 130

    finally:
        _terminate_all(active)
        _deactivate_session_marker(platform)


if __name__ == "__main__":
    sys.exit(main())
