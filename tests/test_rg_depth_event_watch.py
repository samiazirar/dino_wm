import json
import os
import pathlib
import signal
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
WATCHER = ROOT / "tools" / "rg_depth_event_watch.sh"
LAUNCHER = ROOT / "tools" / "launch_rg_depth_event_watch.sh"
JOBS = "26738306,26738307,26738308,26738309"


def job_rows(*states):
    return "".join(
        f"{job}|{state}|0:0|12\n"
        for job, state in zip(JOBS.split(","), states, strict=True)
    )


def run_watcher(tmp_path, *, mode, rows="", message_fails=False, max_polls=0):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ssh").write_text(
        "#!/bin/bash\n"
        "if [ \"$RG_TEST_SSH_MODE\" = fail ]; then\n"
        "  printf 'deliberate ssh failure\\n' >&2\n"
        "  exit 17\n"
        "fi\n"
        "printf '%s' \"$RG_TEST_ROWS\"\n"
    )
    (fake_bin / "herdr-role-message").write_text(
        "#!/bin/bash\n"
        "if [ ! -f \"$RG_TEST_EVENT\" ]; then\n"
        "  printf 'event missing before delivery\\n' >&2\n"
        "  exit 31\n"
        "fi\n"
        "if [ \"${RG_TEST_MESSAGE_FAIL:-0}\" = 1 ]; then\n"
        "  printf 'deliberate delivery failure\\n' >&2\n"
        "  exit 29\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >> \"$RG_TEST_MESSAGE_LOG\"\n"
    )
    for command in fake_bin.iterdir():
        command.chmod(0o755)

    event = tmp_path / "watch.event.json"
    log = tmp_path / "watch.log"
    messages = tmp_path / "messages.log"
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RG_TEST_SSH_MODE": mode,
        "RG_TEST_ROWS": rows,
        "RG_TEST_MESSAGE_FAIL": "1" if message_fails else "0",
        "RG_TEST_MESSAGE_LOG": str(messages),
        "RG_TEST_EVENT": str(event),
        "RG_DEPTH_WATCH_LOG": str(log),
        "INTERVAL_SECONDS": "0",
        "RETRY_LIMIT": "1",
        "RG_DEPTH_WATCH_MAX_POLLS": str(max_polls),
    }
    result = subprocess.run(
        ["bash", str(WATCHER), JOBS, str(event)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert event.is_file()
    assert not list(tmp_path.glob("watch.event.json.tmp.*"))
    return json.loads(event.read_text()), log.read_text(), messages


def test_ssh_failure_writes_atomic_monitor_error(tmp_path):
    receipt, log, _ = run_watcher(tmp_path, mode="fail")

    assert receipt["kind"] == "MONITOR_ERROR"
    assert receipt["reason"] == "SSH_OR_SACCT_FAILED"
    assert receipt["consecutive_failures"] == 1
    assert "POLL_ERROR kind=SSH_OR_SACCT_FAILED" in log


def test_empty_rows_write_atomic_monitor_error(tmp_path):
    receipt, _, _ = run_watcher(tmp_path, mode="rows")

    assert receipt["kind"] == "MONITOR_ERROR"
    assert receipt["reason"].startswith("MISSING_EXPECTED_JOB_ROWS_")
    assert receipt["jobs"] == []


def test_missing_rows_write_atomic_monitor_error(tmp_path):
    rows = job_rows("COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED").rsplit("\n", 2)[0] + "\n"
    receipt, _, _ = run_watcher(tmp_path, mode="rows", rows=rows)

    assert receipt["kind"] == "MONITOR_ERROR"
    assert receipt["reason"] == "MISSING_EXPECTED_JOB_ROWS_26738309"
    assert [row["job_id"] for row in receipt["jobs"]] == JOBS.split(",")[:3]


def test_pending_rows_remain_nonterminal(tmp_path):
    receipt, _, _ = run_watcher(
        tmp_path,
        mode="rows",
        rows=job_rows("PENDING", "RUNNING", "CONFIGURING", "PENDING"),
        max_polls=1,
    )

    assert receipt["kind"] == "MONITOR_ERROR"
    assert receipt["reason"] == "MAX_POLLS_REACHED_WITH_NONTERMINAL_JOBS"
    assert [row["state"] for row in receipt["jobs"]] == [
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "PENDING",
    ]


def test_terminal_success_writes_receipt_before_delivery(tmp_path):
    receipt, log, messages = run_watcher(
        tmp_path,
        mode="rows",
        rows=job_rows("COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"),
    )

    assert receipt["kind"] == "TERMINAL"
    assert receipt["reason"] == "ALL_EXPECTED_JOBS_TERMINAL_failed=0"
    assert messages.read_text().startswith("operations MATERIAL ROPE/GRANULAR DEPTH WATCH EVENT")
    assert "RECEIPT_WRITTEN" in log
    assert "DELIVERY_OK" in log


def test_terminal_failure_marks_failed(tmp_path):
    receipt, _, _ = run_watcher(
        tmp_path,
        mode="rows",
        rows=job_rows("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"),
    )

    assert receipt["kind"] == "TERMINAL"
    assert receipt["reason"] == "ALL_EXPECTED_JOBS_TERMINAL_failed=1"


def test_message_failure_preserves_terminal_receipt(tmp_path):
    receipt, log, messages = run_watcher(
        tmp_path,
        mode="rows",
        rows=job_rows("COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"),
        message_fails=True,
    )

    assert receipt["kind"] == "TERMINAL"
    assert not messages.exists()
    assert "RECEIPT_WRITTEN" in log
    assert "DELIVERY_ERROR kind=TERMINAL" in log


def test_foreground_launcher_survives_an_interval_and_cleans_temporary_state(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ssh").write_text(
        "#!/bin/bash\n"
        f"printf '%s' '{job_rows('PENDING', 'PENDING', 'PENDING', 'PENDING')}'\n"
    )
    (fake_bin / "herdr-role-message").write_text("#!/bin/bash\nexit 0\n")
    for command in fake_bin.iterdir():
        command.chmod(0o755)

    event = tmp_path / "watch.event.json"
    pid_file = tmp_path / "watch.pid"
    log = tmp_path / "watch.log"
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "INTERVAL_SECONDS": "1",
        "RETRY_LIMIT": "3",
    }
    process = subprocess.Popen(
        ["bash", str(LAUNCHER), JOBS, str(event), str(pid_file), str(log)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists()
    watcher_pid = int(pid_file.read_text())
    try:
        assert watcher_pid == process.pid
        time.sleep(1.3)
        os.kill(watcher_pid, 0)
        assert log.read_text().count("POLL_OK kind=NONTERMINAL") >= 2

        os.kill(watcher_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not event.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        receipt = json.loads(event.read_text())
        assert receipt["kind"] == "MONITOR_ERROR"
        assert receipt["reason"] == "WATCHER_SIGNAL_TERM"
        assert not pid_file.exists()
        assert not list(tmp_path.glob("watch.event.json.snapshot.*"))
        assert not list(tmp_path.glob("watch.event.json.parsed.*"))
    finally:
        try:
            os.kill(watcher_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
