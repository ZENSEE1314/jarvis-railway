"""Local Mark-style tools for Desktop JARVIS.

Inspired by FatihMakes/Mark-XXXIX-OR's action catalog, but implemented with
standard-library Windows-safe handlers for this project.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import webbrowser
from pathlib import Path


APP_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "firefox": "firefox",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "terminal": "cmd.exe",
    "powershell": "powershell.exe",
    "vscode": "code",
    "visual studio code": "code",
    "whatsapp": "WhatsApp",
    "spotify": "Spotify",
    "discord": "Discord",
    "telegram": "Telegram",
    "zoom": "Zoom",
    "steam": "steam",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
}


FILE_TYPE_MAP = {
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico"},
    "Documents": {".pdf", ".doc", ".docx", ".txt", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".md"},
    "Videos": {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm"},
    "Music": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz"},
    "Code": {".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".java", ".cpp", ".cs", ".go", ".rs"},
}


def resolve_path(raw: str) -> Path:
    shortcuts = {
        "desktop": Path.home() / "Desktop",
        "downloads": Path.home() / "Downloads",
        "documents": Path.home() / "Documents",
        "pictures": Path.home() / "Pictures",
        "music": Path.home() / "Music",
        "videos": Path.home() / "Videos",
        "home": Path.home(),
    }
    key = raw.strip().strip('"').lower()
    if key in shortcuts:
        return shortcuts[key]
    return Path(os.path.expandvars(os.path.expanduser(raw.strip().strip('"'))))


def format_size(value: int) -> str:
    amount = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if amount < 1024:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} PB"


def normalize_app(name: str) -> str:
    key = name.lower().strip()
    if key in APP_ALIASES:
        return APP_ALIASES[key]
    for alias, command in APP_ALIASES.items():
        if alias in key or key in alias:
            return command
    return name.strip()


def open_target(target: str) -> str:
    target = target.strip()
    if not target:
        return "Tell me what to open."
    if re.match(r"^https?://", target, re.I):
        webbrowser.open(target)
        return f"Opened URL: {target}"
    path = resolve_path(target)
    if path.exists():
        os.startfile(str(path))  # type: ignore[attr-defined]
        return f"Opened: {path}"
    command = normalize_app(target)
    try:
        subprocess.Popen(command, shell=True)
        return f"Launched: {target}"
    except Exception as exc:
        return f"I could not open {target}: {exc}"


def list_files(location: str = "desktop", limit: int = 30) -> str:
    path = resolve_path(location)
    if not path.exists():
        return f"Path not found: {path}"
    if not path.is_dir():
        return f"That is not a folder: {path}"
    rows = []
    for item in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))[:limit]:
        try:
            if item.is_dir():
                rows.append(f"[folder] {item.name}")
            else:
                rows.append(f"[file] {item.name} ({format_size(item.stat().st_size)})")
        except OSError:
            continue
    if not rows:
        return f"{path} is empty."
    return f"Contents of {path}:\n" + "\n".join(rows)


def find_files(query: str, location: str = "documents", limit: int = 20) -> str:
    path = resolve_path(location)
    if not path.exists() or not path.is_dir():
        return f"Search folder not found: {path}"
    query = query.strip().lower()
    results = []
    for item in path.rglob("*"):
        if item.is_file() and query in item.name.lower():
            try:
                results.append(f"{item.name} ({format_size(item.stat().st_size)}) - {item.parent}")
            except OSError:
                results.append(f"{item.name} - {item.parent}")
        if len(results) >= limit:
            break
    if not results:
        return f"I could not find files matching '{query}' in {path}."
    return "Found files:\n" + "\n".join(results)


def read_file(path_text: str, max_chars: int = 3000) -> str:
    path = resolve_path(path_text)
    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"That is not a file: {path}"
    content = path.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... truncated from {len(content)} characters."
    return content


def write_file(path_text: str, content: str) -> str:
    path = resolve_path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Written to: {path}"


def create_folder(path_text: str) -> str:
    path = resolve_path(path_text)
    path.mkdir(parents=True, exist_ok=True)
    return f"Folder created: {path}"


def disk_usage(location: str = "home") -> str:
    path = resolve_path(location)
    usage = shutil.disk_usage(path)
    return (
        f"Disk usage for {path}:\n"
        f"Total: {format_size(usage.total)}\n"
        f"Used: {format_size(usage.used)}\n"
        f"Free: {format_size(usage.free)}"
    )


def organize_desktop() -> str:
    desktop = resolve_path("desktop")
    if not desktop.exists():
        return f"Desktop not found: {desktop}"
    moved = []
    for item in desktop.iterdir():
        if item.is_dir() or item.suffix.lower() in {".lnk", ".url"}:
            continue
        folder_name = "Others"
        for group, extensions in FILE_TYPE_MAP.items():
            if item.suffix.lower() in extensions:
                folder_name = group
                break
        target_dir = desktop / folder_name
        target_dir.mkdir(exist_ok=True)
        target = target_dir / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
        moved.append(f"{item.name} -> {folder_name}")
    if not moved:
        return "Desktop is already organized."
    return f"Desktop organized. Moved {len(moved)} file(s):\n" + "\n".join(moved[:12])


def handle_mark_tool(text: str) -> tuple[str, bool]:
    lowered = text.lower().strip()

    if re.match(r"^(open|launch|start)\s+", lowered):
        target = re.sub(r"^(open|launch|start)\s+", "", text, flags=re.I).strip()
        return open_target(target), True

    if any(phrase in lowered for phrase in ["list desktop", "show desktop files", "what is on desktop"]):
        return list_files("desktop"), True
    if any(phrase in lowered for phrase in ["list downloads", "show downloads"]):
        return list_files("downloads"), True
    if any(phrase in lowered for phrase in ["list documents", "show documents"]):
        return list_files("documents"), True

    find_match = re.search(r"\bfind\s+(?:file|files)?\s*(?:called|named)?\s*(.+)", text, re.I)
    if find_match:
        query = find_match.group(1).strip()
        return find_files(query), True

    read_match = re.search(r"\bread\s+(?:file\s+)?(.+)", text, re.I)
    if read_match:
        return read_file(read_match.group(1)), True

    folder_match = re.search(r"\bcreate\s+(?:folder|directory)\s+(.+)", text, re.I)
    if folder_match:
        return create_folder(folder_match.group(1)), True

    write_match = re.search(r"\bwrite\s+(.+?)\s+to\s+file\s+(.+)", text, re.I)
    if write_match:
        return write_file(write_match.group(2), write_match.group(1)), True

    if "organize desktop" in lowered:
        return organize_desktop(), True
    if "disk usage" in lowered or "storage usage" in lowered or "free space" in lowered:
        return disk_usage("home"), True

    return "", False
