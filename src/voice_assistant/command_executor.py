"""
Command Executor for NovaMaster Voice Assistant.

Intercepts voice commands and executes actions (open URLs, search, control services)
instead of just chatting. Runs BEFORE sending to LLM so common actions are instant.
"""

import logging
import re
import subprocess
import sys
import shutil
from typing import Optional

# ── Action Definitions ──────────────────────────────────────────────────────

BROWSER_ACTIONS = {
    "youtube": "https://youtube.com",
    "google": "https://google.com",
    "github": "https://github.com",
    "x": "https://x.com",
    "twitter": "https://x.com",
    "reddit": "https://reddit.com",
    "gmail": "https://mail.google.com",
    "maps": "https://maps.google.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
}

SERVICE_ACTIONS = {
    "cockpit": "http://127.0.0.1:8888",
    "jarvis": "http://127.0.0.1:8888",
    "office": "http://127.0.0.1:3000",
    "grafana": "http://127.0.0.1:3001",
    "comfyui": "http://127.0.0.1:8188",
    "n8n": "http://127.0.0.1:5678",
    "open webui": "http://127.0.0.1:3080",
    "webui": "http://127.0.0.1:3080",
    "hermes": "http://127.0.0.1:9119",
    "uptime kuma": "http://127.0.0.1:3002/dashboard",
    "uptime": "http://127.0.0.1:3002/dashboard",
    "portainer": "http://127.0.0.1:9000",
    "qdrant": "http://127.0.0.1:6333/dashboard",
}

# Open patterns: "open youtube", "open youtube.com", "open the office"
OPEN_RE = re.compile(
    r"(?:open| launch| start| ga naar| go to)\s+"
    r"(?:the\s+|mijn\s+|de\s+)?"
    r"(.+?)$",
    re.IGNORECASE,
)

SEARCH_RE = re.compile(
    r"(?:search|zoek|google|zoek op)\s+(?:for|naar|op|to)?\s*(.+)",
    re.IGNORECASE,
)

PLAY_RE = re.compile(
    r"(?:play|speel|spelen)\s+(.+?)(?:\s+(?:on|op)\s+(.+))?$",
    re.IGNORECASE,
)


def _open_url(url: str) -> bool:
    """Open URL in Windows browser from WSL. Avoids transparent/frameless window issue."""
    try:
        # WSL: use cmd.exe /c start to open in Windows default browser
        if shutil.which("cmd.exe"):
            subprocess.run(["cmd.exe", "/c", "start", "", url], capture_output=True, timeout=5)
            return True
        # macOS
        if sys.platform == "darwin":
            subprocess.run(["open", url], capture_output=True, timeout=5)
            return True
        # Linux native
        subprocess.run(["xdg-open", url], capture_output=True, timeout=5)
        return True
    except Exception as e:
        logging.error(f"Failed to open URL {url}: {e}")
        return False


def _resolve_target(name: str) -> Optional[str]:
    """Resolve a spoken name to a URL."""
    name = name.strip().lower()
    # Strip .com/.nl etc
    clean = re.sub(r"\.(com|nl|org|io|net|app)$", "", name)

    # Check browser actions first
    for key, url in BROWSER_ACTIONS.items():
        if key in clean or clean in key:
            return url

    # Check service actions
    for key, url in SERVICE_ACTIONS.items():
        if key in clean or clean in key:
            return url

    # Try as direct URL
    if "." in clean and " " not in clean:
        return f"https://{clean}.com"

    return None


def execute_open(text: str) -> Optional[str]:
    """Handle 'open X' commands. Returns spoken confirmation or None."""
    m = OPEN_RE.search(text)
    if not m:
        return None

    target = m.group(1).strip()
    url = _resolve_target(target)

    if url:
        if _open_url(url):
            logging.info(f"Opened browser: {url}")
            return f"Opening {target}"
        else:
            return f"Sorry, could not open {target}"
    else:
        # Try as unknown website
        try_url = f"https://{target.lower().replace(' ', '')}.com"
        if _open_url(try_url):
            logging.info(f"Opened browser: {try_url}")
            return f"Opening {target}"
        else:
            return None


def execute_search(text: str) -> Optional[str]:
    """Handle 'search for X' commands."""
    m = SEARCH_RE.search(text)
    if not m:
        return None

    query = m.group(1).strip()
    if not query:
        return None

    try:
        encoded = query.replace(" ", "+")
        url = f"https://google.com/search?q={encoded}"
        _open_url(url)
        logging.info(f"Searched Google: {query}")
        return f"Searching for {query}"
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return None


def execute_system(text: str) -> Optional[str]:
    """Handle system commands."""
    t = text.lower().strip()

    # Volume control
    if "volume up" in t or "harder" in t:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], capture_output=True)
        return "Volume up"
    if "volume down" in t or "zachter" in t:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], capture_output=True)
        return "Volume down"
    if "mute" in t or "geluid uit" in t:
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], capture_output=True)
        return "Muted"
    if "unmute" in t or "geluid aan" in t:
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], capture_output=True)
        return "Unmuted"

    # Screenshot
    if "screenshot" in t or "schermafbeelding" in t:
        try:
            subprocess.run(["cmd.exe", "/c", "start", "snippingtool", "/clip"], capture_output=True, timeout=3)
        except Exception:
            subprocess.Popen(["gnome-screenshot", "-f", "/tmp/voice_screenshot.png"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "Taking screenshot"

    return None


COMMAND_HANDLERS = [
    ("open", execute_open),
    ("search", execute_search),
    ("system", execute_system),
]


def execute_command(text: str) -> Optional[str]:
    """
    Try to execute a voice command. Returns spoken response if handled, None to fall through to LLM.
    """
    # Try specific handlers based on keywords
    for handler_name, handler in COMMAND_HANDLERS:
        try:
            result = handler(text)
            if result:
                return result
        except Exception as e:
            logging.error(f"Handler {handler_name} failed: {e}")

    return None
