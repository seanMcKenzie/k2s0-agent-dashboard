#!/usr/bin/env python3
"""
agent-status-server.py
K2S0 Agent Dashboard - Local Status Server
Runs on port 7800, serves GET /agents with live workspace data.

Start: python3 agent-status-server.py
"""

import json
import os
import re
import sys
import time
import glob
import uuid
import threading
import collections
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

OPENCLAW_JSON = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def load_agent_models():
    """Load per-agent model from openclaw.json, falling back to global default."""
    models = {}
    try:
        with open(OPENCLAW_JSON, "r", encoding="utf-8") as f:
            config = json.load(f)
        global_model = (
            config.get("agents", {})
            .get("defaults", {})
            .get("model", {})
            .get("primary", DEFAULT_MODEL)
        )
        for agent in config.get("agents", {}).get("list", []):
            aid = agent.get("id")
            if aid:
                models[aid] = agent.get("model", global_model)
        models["__default__"] = global_model
    except Exception:
        models["__default__"] = DEFAULT_MODEL
    return models


AGENT_MODELS = load_agent_models()

# ============================================================
# CONFIG
# ============================================================
PORT = 7800
ACTIVE_THRESHOLD_SECONDS = 300  # 5 minutes
ACTIVITY_LOG_MAX = 50
POLL_INTERVAL = 5  # seconds - background watcher refresh
LOG_RETENTION_DAYS = 7

AGENTS = [
    {"id": "main",      "name": "K2S0",      "role": "Coordinator", "emoji": "🤖"},
    {"id": "developer", "name": "Charlie",   "role": "Developer",   "emoji": "👨‍💻"},
    {"id": "pm",        "name": "Dennis",    "role": "PM",          "emoji": "📋"},
    {"id": "qa",        "name": "Mac",       "role": "QA",          "emoji": "🔍"},
    {"id": "devops",    "name": "Frank",     "role": "DevOps",      "emoji": "🔧"},
    {"id": "research",  "name": "Sweet Dee", "role": "Research",    "emoji": "🔬"},
    {"id": "designer",  "name": "Cricket",   "role": "Designer",    "emoji": "🎨"},
]

WORKSPACE_PATTERNS = {
    "main":      ["workspace", "workspace-main", "workspace-coordinator"],
    "developer": ["workspace-developer", "workspace-dev", "workspace-charlie"],
    "pm":        ["workspace-pm", "workspace-dennis"],
    "qa":        ["workspace-qa", "workspace-mac"],
    "devops":    ["workspace-devops", "workspace-frank"],
    "research":  ["workspace-research", "workspace-sweetdee", "workspace-sweet-dee"],
    "designer":  ["workspace-designer", "workspace-cricket"],
}

OPENCLAW_BASE = Path.home() / ".openclaw"
LOGS_DIR = OPENCLAW_BASE / "workspace" / "logs"
ACTIVITY_JSONL = LOGS_DIR / "activity.jsonl"

# ============================================================
# STATE (thread-safe via lock)
# ============================================================
lock = threading.Lock()
agent_cache = {}
activity_log = collections.deque(maxlen=ACTIVITY_LOG_MAX)
file_mtime_cache = {}
file_size_cache = {}
file_linecount_cache = {}
agent_last_active = {}

IDLE_NOTIFY_THRESHOLD = 600  # 10 minutes


# ============================================================
# JSONL LOG FILE HELPERS
# ============================================================
def ensure_logs_dir():
    """Create the logs directory if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def append_log_entry(entry: dict):
    """Append a single log entry to the JSONL file."""
    ensure_logs_dir()
    try:
        with open(ACTIVITY_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log] ERROR writing log entry: {e}")


def read_log_entries() -> list:
    """Read all log entries from the JSONL file."""
    if not ACTIVITY_JSONL.exists():
        return []
    entries = []
    try:
        with open(ACTIVITY_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"[log] ERROR reading log entries: {e}")
    return entries


def prune_log_entries():
    """Remove log entries older than LOG_RETENTION_DAYS days. Rewrites the file."""
    if not ACTIVITY_JSONL.exists():
        return
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)
    entries = read_log_entries()
    kept = []
    pruned = 0
    for entry in entries:
        ts_str = entry.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= cutoff:
                kept.append(entry)
            else:
                pruned += 1
        except Exception:
            kept.append(entry)  # keep entries with unparseable timestamps
    if pruned > 0:
        ensure_logs_dir()
        with open(ACTIVITY_JSONL, "w", encoding="utf-8") as f:
            for entry in kept:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[log] Pruned {pruned} entries older than {LOG_RETENTION_DAYS} days. Kept {len(kept)}.")


def get_existing_log_ids() -> set:
    """Return set of existing log entry IDs to avoid duplicates."""
    entries = read_log_entries()
    return {e.get("id") for e in entries if e.get("id")}


def make_log_entry(
    agent: str,
    agent_name: str,
    event_type: str,
    title: str,
    detail: str = "",
    file: str = "",
    model: str = "",
    estimated_tokens: int = 0,
    tags: list = None,
    timestamp: str = None,
) -> dict:
    """Create a structured log entry."""
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
    if not model:
        model = AGENT_MODELS.get(agent, AGENT_MODELS.get("__default__", DEFAULT_MODEL))
    return {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "agent": agent,
        "agent_name": agent_name,
        "event_type": event_type,
        "title": title,
        "detail": detail,
        "file": file,
        "model": model,
        "estimated_tokens": estimated_tokens,
        "tags": tags or [],
    }


# ============================================================
# MEMORY FILE IMPORT (bootstrap)
# ============================================================
def extract_h2_sections(content: str) -> list:
    """Extract H2 sections from markdown content as (title, body) tuples."""
    sections = []
    current_title = None
    current_body = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)

    if current_title is not None:
        sections.append((current_title, "\n".join(current_body).strip()))

    return sections


def infer_tags_from_content(title: str, body: str) -> list:
    """Infer tags from section title and body text."""
    text = (title + " " + body).lower()
    tags = []
    tag_keywords = {
        "memory": ["memory", "remember", "notes"],
        "research": ["research", "findings", "study", "data"],
        "figma": ["figma"],
        "github": ["github", "repo", "push", "commit", "pr"],
        "docker": ["docker", "container", "deploy"],
        "spring": ["spring boot", "spring", "java"],
        "discord": ["discord"],
        "api": ["api", "endpoint", "rest"],
        "design": ["design", "wireframe", "ui", "ux"],
        "voice": ["voice", "tts", "audio"],
        "testing": ["test", "qa", "bug", "fix"],
        "task": ["task", "todo", "backlog"],
    }
    for tag, keywords in tag_keywords.items():
        if any(k in text for k in keywords):
            tags.append(tag)
    return tags


def import_memory_files():
    """Scan all agent memory files from past 7 days and import as log entries."""
    print("[log] Importing memory files from past 7 days...")
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)
    existing_ids = get_existing_log_ids()

    # Track which (agent, file, section_title) combos we've already imported
    # We use a content hash approach: id is deterministic based on content
    imported_count = 0

    for agent in AGENTS:
        agent_id = agent["id"]
        agent_name = agent["name"]
        workspace = find_workspace(agent_id)
        if workspace is None:
            continue

        memory_dir = workspace / "memory"
        if not memory_dir.is_dir():
            continue

        for md_file in sorted(memory_dir.glob("*.md")):
            try:
                stat = md_file.stat()
                file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                # Only import files modified in the past 7 days
                if file_mtime < cutoff:
                    continue

                content = md_file.read_text(encoding="utf-8", errors="ignore")
                sections = extract_h2_sections(content)
                rel_path = f"memory/{md_file.name}"

                if not sections:
                    # Import the whole file as one entry
                    entry_key = f"{agent_id}:{rel_path}:full"
                    # Use a deterministic ID based on content signature
                    det_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_key + content[:100]))
                    if det_id not in existing_ids:
                        title = md_file.stem  # e.g. "2026-02-27"
                        snippet = content[:500]
                        entry = make_log_entry(
                            agent=agent_id,
                            agent_name=agent_name,
                            event_type="memory_update",
                            title=f"Memory: {title}",
                            detail=snippet,
                            file=rel_path,
                            estimated_tokens=len(content) // 4,
                            tags=infer_tags_from_content(title, content),
                            timestamp=file_mtime.isoformat(),
                        )
                        entry["id"] = det_id
                        append_log_entry(entry)
                        existing_ids.add(det_id)
                        imported_count += 1
                else:
                    for (section_title, section_body) in sections:
                        entry_key = f"{agent_id}:{rel_path}:{section_title}"
                        det_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_key + section_body[:50]))
                        if det_id not in existing_ids:
                            tags = infer_tags_from_content(section_title, section_body)
                            entry = make_log_entry(
                                agent=agent_id,
                                agent_name=agent_name,
                                event_type="memory_update",
                                title=section_title,
                                detail=section_body[:800],
                                file=rel_path,
                                estimated_tokens=len(section_body) // 4,
                                tags=tags,
                                timestamp=file_mtime.isoformat(),
                            )
                            entry["id"] = det_id
                            append_log_entry(entry)
                            existing_ids.add(det_id)
                            imported_count += 1

            except Exception as e:
                print(f"[log] Error importing {md_file}: {e}")

    print(f"[log] Import complete. Added {imported_count} new entries.")
    return imported_count


# ============================================================
# WORKSPACE SCANNING
# ============================================================
def find_workspace(agent_id):
    patterns = WORKSPACE_PATTERNS.get(agent_id, [])
    for pattern in patterns:
        candidate = OPENCLAW_BASE / pattern
        if candidate.is_dir():
            return candidate
    return None


def scan_md_files(workspace_dir):
    if workspace_dir is None:
        return []
    results = []
    for md_file in workspace_dir.rglob("*.md"):
        try:
            stat = md_file.stat()
            results.append((md_file, stat.st_mtime, stat.st_size))
        except OSError:
            pass
    return results


def read_last_task(workspace_dir):
    if workspace_dir is None:
        return None
    memory_dir = workspace_dir / "memory"
    if not memory_dir.is_dir():
        return None
    memory_files = sorted(
        memory_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for mf in memory_files[:3]:
        try:
            content = mf.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("-") or line.startswith("*"):
                    line = line.lstrip("-* ").strip()
                if len(line) > 10:
                    return line[:120]
        except Exception:
            pass
    return None


def get_total_workspace_bytes(workspace_dir):
    if workspace_dir is None:
        return 0
    total = 0
    for md_file in workspace_dir.rglob("*.md"):
        try:
            total += md_file.stat().st_size
        except OSError:
            pass
    return total


def get_total_workspace_chars(workspace_dir):
    if workspace_dir is None:
        return 0
    total = 0
    for md_file in workspace_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            total += len(content)
        except OSError:
            pass
    return total


def read_file_snippet(path, n=5):
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        tail = lines[-n:] if len(lines) >= n else lines
        return " · ".join(tail)[:300]
    except Exception:
        return ""


def count_file_lines(path):
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        return sum(1 for l in content.splitlines() if l.strip())
    except Exception:
        return 0


def count_memory_events(workspace_dir):
    if workspace_dir is None:
        return 0
    memory_dir = workspace_dir / "memory"
    if not memory_dir.is_dir():
        return 0
    return len(list(memory_dir.glob("*.md")))


# ============================================================
# AGENT STATUS BUILD
# ============================================================
def build_agent_status(agent):
    agent_id   = agent["id"]
    workspace  = find_workspace(agent_id)
    md_files   = scan_md_files(workspace)

    now = time.time()
    last_mtime = None
    last_file  = None

    for (path, mtime, _size) in md_files:
        if last_mtime is None or mtime > last_mtime:
            last_mtime = mtime
            last_file  = path

    status = "idle"
    last_seen_iso = None

    if last_mtime is not None:
        age = now - last_mtime
        if age <= ACTIVE_THRESHOLD_SECONDS:
            status = "active"
        last_seen_iso = datetime.fromtimestamp(last_mtime, tz=timezone.utc).isoformat()

    workspace_bytes = get_total_workspace_bytes(workspace)
    workspace_chars = get_total_workspace_chars(workspace)
    last_task       = read_last_task(workspace)
    event_count     = count_memory_events(workspace)

    model_full  = AGENT_MODELS.get(agent_id, AGENT_MODELS.get("__default__", DEFAULT_MODEL))
    model_short = model_full.split("/", 1)[-1] if "/" in model_full else model_full

    return {
        "id":               agent_id,
        "name":             agent["name"],
        "role":             agent["role"],
        "emoji":            agent["emoji"],
        "status":           status,
        "last_seen":        last_seen_iso,
        "last_task":        last_task,
        "workspace_bytes":  workspace_bytes,
        "workspace_chars":  workspace_chars,
        "workspace_path":   str(workspace) if workspace else None,
        "event_count":      event_count,
        "model":            model_full,
        "model_short":      model_short,
        "estimated_tokens": round(workspace_chars / 4),
    }


# ============================================================
# FILE WATCHER (background thread)
# ============================================================
def detect_file_changes(agent_id, workspace):
    global file_mtime_cache, file_size_cache, file_linecount_cache, agent_last_active
    if workspace is None:
        return

    now = time.time()
    md_files = scan_md_files(workspace)
    agent_meta = next((a for a in AGENTS if a["id"] == agent_id), {})

    for (path, mtime, cur_size) in md_files:
        path_str   = str(path)
        prev_mtime = file_mtime_cache.get(path_str)
        prev_size  = file_size_cache.get(path_str)
        prev_lines = file_linecount_cache.get(path_str)

        is_new     = prev_mtime is None
        is_changed = (not is_new) and mtime > prev_mtime

        if is_new or is_changed:
            cur_lines  = count_file_lines(path)
            size_delta = cur_size - (prev_size if prev_size is not None else 0)
            line_delta = cur_lines - (prev_lines if prev_lines is not None else 0)
            snippet    = read_file_snippet(path, n=5)

            try:
                rel = str(path.relative_to(workspace))
            except ValueError:
                rel = path.name

            if is_new:
                event_type   = "task"
                severity     = "task"
                event_detail = f"Created {rel}"
                log_event_type = "file_change"
                log_title = f"New file: {rel}"
            else:
                event_type   = "updated"
                severity     = "info"
                delta_str    = (f"+{line_delta}" if line_delta >= 0 else str(line_delta)) + " lines"
                event_detail = f"Updated {rel} ({delta_str})"
                log_event_type = "file_change"
                log_title = f"Updated: {rel} ({delta_str})"

            ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            event = {
                "agent":        agent_id,
                "name":         agent_meta.get("name", agent_id),
                "emoji":        agent_meta.get("emoji", "🤖"),
                "timestamp":    ts,
                "file_changed": rel,
                "event_type":   event_type,
                "event_detail": event_detail,
                "snippet":      snippet,
                "size_delta":   size_delta,
                "severity":     severity,
            }
            with lock:
                activity_log.appendleft(event)
            print(f"[{ts}] {severity.upper():6s}: {agent_meta.get('name', agent_id)} → {event_detail}")

            # Also append to persistent JSONL log
            log_entry = make_log_entry(
                agent=agent_id,
                agent_name=agent_meta.get("name", agent_id),
                event_type=log_event_type,
                title=log_title,
                detail=snippet,
                file=rel,
                estimated_tokens=cur_size // 4,
                tags=infer_tags_from_content(rel, snippet),
                timestamp=ts,
            )
            append_log_entry(log_entry)

            file_linecount_cache[path_str] = cur_lines
            agent_last_active[agent_id] = now

        file_mtime_cache[path_str] = mtime
        file_size_cache[path_str]  = cur_size

    # Idle event
    last_active = agent_last_active.get(agent_id)
    if last_active is not None and (now - last_active) >= IDLE_NOTIFY_THRESHOLD:
        ts = datetime.now(tz=timezone.utc).isoformat()
        idle_event = {
            "agent":        agent_id,
            "name":         agent_meta.get("name", agent_id),
            "emoji":        agent_meta.get("emoji", "🤖"),
            "timestamp":    ts,
            "file_changed": "",
            "event_type":   "idle",
            "event_detail": f"{agent_meta.get('name', agent_id)} has been idle for {int((now - last_active) // 60)}+ min",
            "snippet":      "",
            "size_delta":   0,
            "severity":     "idle",
        }
        with lock:
            activity_log.appendleft(idle_event)
        print(f"[{ts}] IDLE  : {agent_meta.get('name', agent_id)} → no changes in {int((now - last_active) // 60)}+ min")
        agent_last_active[agent_id] = None


def prune_loop():
    """Hourly pruning of old log entries."""
    time.sleep(3600)
    while True:
        prune_log_entries()
        time.sleep(3600)


def watcher_loop():
    # Prime caches on first run
    for agent in AGENTS:
        workspace = find_workspace(agent["id"])
        if workspace:
            for (path, mtime, size) in scan_md_files(workspace):
                p = str(path)
                file_mtime_cache[p]     = mtime
                file_size_cache[p]      = size
                file_linecount_cache[p] = count_file_lines(path)

    while True:
        time.sleep(POLL_INTERVAL)
        new_cache = {}
        for agent in AGENTS:
            workspace = find_workspace(agent["id"])
            detect_file_changes(agent["id"], workspace)
            status = build_agent_status(agent)
            new_cache[agent["id"]] = status

        with lock:
            agent_cache.update(new_cache)


def initial_load():
    ts = datetime.now(tz=timezone.utc).isoformat()
    for agent in AGENTS:
        status = build_agent_status(agent)
        agent_cache[agent["id"]] = status

        boot_event = {
            "agent":        agent["id"],
            "name":         agent["name"],
            "emoji":        agent["emoji"],
            "timestamp":    ts,
            "file_changed": "",
            "event_type":   "boot",
            "event_detail": f"{agent['name']} agent online · workspace {'found' if status['workspace_path'] else 'not found'}",
            "snippet":      "",
            "size_delta":   0,
            "severity":     "info",
        }
        activity_log.appendleft(boot_event)

        # Log boot to JSONL
        boot_log = make_log_entry(
            agent=agent["id"],
            agent_name=agent["name"],
            event_type="boot",
            title=f"{agent['name']} agent online",
            detail=f"Workspace {'found' if status['workspace_path'] else 'not found'} at {status['workspace_path']}",
            timestamp=ts,
            tags=["boot"],
        )
        append_log_entry(boot_log)

    print(f"[boot] Loaded {len(agent_cache)} agents.")


# ============================================================
# LOG QUERY HELPERS
# ============================================================
def query_logs(agent: str = None, search: str = None, date: str = None,
               event_type: str = None, limit: int = 50, offset: int = 0) -> list:
    """Read and filter log entries."""
    entries = read_log_entries()

    # Filter
    if agent:
        entries = [e for e in entries if e.get("agent") == agent]
    if event_type:
        entries = [e for e in entries if e.get("event_type") == event_type]
    if date:
        entries = [e for e in entries if e.get("timestamp", "").startswith(date)]
    if search:
        search_lower = search.lower()
        def matches(e):
            return (
                search_lower in (e.get("title") or "").lower() or
                search_lower in (e.get("detail") or "").lower() or
                search_lower in (e.get("agent_name") or "").lower() or
                search_lower in " ".join(e.get("tags") or []).lower()
            )
        entries = [e for e in entries if matches(e)]

    # Sort newest first
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    # Paginate
    return entries[offset: offset + limit]


def get_log_agents() -> list:
    """Return list of unique agents in the log."""
    entries = read_log_entries()
    seen = {}
    for e in entries:
        aid = e.get("agent")
        if aid and aid not in seen:
            seen[aid] = e.get("agent_name", aid)
    return [{"agent": k, "agent_name": v} for k, v in seen.items()]


def get_log_summary() -> dict:
    """Count events per agent per day for the past 7 days."""
    entries = read_log_entries()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)

    # Build structure: {date: {agent: count}}
    summary = {}
    for e in entries:
        ts_str = e.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts < cutoff:
                continue
            day = ts.strftime("%Y-%m-%d")
            agent = e.get("agent", "unknown")
            if day not in summary:
                summary[day] = {}
            summary[day][agent] = summary[day].get(agent, 0) + 1
        except Exception:
            pass

    return summary


def get_total_log_count() -> int:
    """Return total number of log entries."""
    return len(read_log_entries())


# ============================================================
# HTTP SERVER
# ============================================================
class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if "404" in str(args) or "500" in str(args):
            super().log_message(fmt, *args)

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        def qs_get(key, default=None):
            vals = qs.get(key, [])
            return vals[0] if vals else default

        if path == "/agents":
            self.handle_agents()
        elif path == "/activity":
            self.handle_activity()
        elif path == "/health":
            self.handle_health()
        elif path == "/logs":
            self.handle_logs(qs_get, qs)
        elif path == "/logs/agents":
            self.handle_logs_agents()
        elif path == "/logs/summary":
            self.handle_logs_summary()
        else:
            self.send_response(404)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not found"}).encode())

    def handle_agents(self):
        with lock:
            data = list(agent_cache.values())
        self.send_json(data)

    def handle_activity(self):
        with lock:
            data = list(activity_log)
        self.send_json(data)

    def handle_health(self):
        self.send_json({
            "status": "ok",
            "agents": len(agent_cache),
            "uptime": int(time.time() - START_TIME),
            "activity_events": len(activity_log),
            "log_entries": get_total_log_count(),
        })

    def handle_logs(self, qs_get, qs):
        agent      = qs_get("agent")
        search     = qs_get("search")
        date       = qs_get("date")
        event_type = qs_get("event_type")
        limit      = int(qs_get("limit", "50"))
        offset     = int(qs_get("offset", "0"))

        limit = min(limit, 200)  # cap at 200

        entries = query_logs(
            agent=agent,
            search=search,
            date=date,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
        self.send_json({
            "entries": entries,
            "count": len(entries),
            "limit": limit,
            "offset": offset,
            "total": get_total_log_count(),
        })

    def handle_logs_agents(self):
        self.send_json(get_log_agents())

    def handle_logs_summary(self):
        self.send_json(get_log_summary())

    def send_json(self, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ============================================================
# MAIN
# ============================================================
START_TIME = time.time()

if __name__ == "__main__":
    print("=" * 56)
    print("  K2S0 Agent Status Server")
    print(f"  Listening on http://localhost:{PORT}")
    print(f"  Endpoints: /agents  /activity  /health")
    print(f"  Log endpoints: /logs  /logs/agents  /logs/summary")
    print(f"  Scanning:  {OPENCLAW_BASE}")
    print("=" * 56)

    # Ensure logs directory exists
    ensure_logs_dir()

    # Prune old entries on startup
    prune_log_entries()

    # Import memory files from past 7 days
    import_memory_files()

    # Initial synchronous load
    initial_load()

    # Print discovered workspaces
    for agent in AGENTS:
        ws = find_workspace(agent["id"])
        ws_str = str(ws) if ws else "(not found)"
        print(f"  {agent['emoji']}  {agent['name']:10s} → {ws_str}")
    print()

    # Start background watcher
    t = threading.Thread(target=watcher_loop, daemon=True)
    t.start()

    # Start hourly prune loop
    p = threading.Thread(target=prune_loop, daemon=True)
    p.start()

    # Log total entries
    total = get_total_log_count()
    print(f"  📝 Activity log: {total} entries in {ACTIVITY_JSONL}")
    print()

    server = HTTPServer(("", PORT), DashboardHandler)
    try:
        print(f"Server running. Press Ctrl+C to stop.\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
        sys.exit(0)
