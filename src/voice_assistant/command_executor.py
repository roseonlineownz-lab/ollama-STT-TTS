"""
NovaMaster Command Executor — Handles voice commands like "open AionUI", "start ComfyUI", etc.
Intercepts LLM responses containing [ACTION:xxx] tags and executes them.
"""
import logging
import subprocess
import webbrowser
import os
import json
import re

logger = logging.getLogger("nova-command")

# Service registry with URLs and start commands
SERVICES = {
    # VPS Services
    "aion ui": {"url": "http://72.62.237.198:25808", "type": "browser"},
    "aionui": {"url": "http://72.62.237.198:25808", "type": "browser"},
    "open webui": {"url": "http://72.62.237.198:3080", "type": "browser"},
    "webui": {"url": "http://72.62.237.198:3080", "type": "browser"},
    "hermes": {"url": "http://72.62.237.198:9119", "type": "browser"},
    "hermes office": {"url": "http://72.62.237.198:9119", "type": "browser"},
    "n8n": {"url": "http://72.62.237.198:5678", "type": "browser"},
    "space agent": {"url": "http://72.62.237.198:3003", "type": "browser"},
    "spaceagent": {"url": "http://72.62.237.198:3003", "type": "browser"},
    "grafana": {"url": "http://72.62.237.198:3001", "type": "browser"},
    "portainer": {"url": "http://72.62.237.198:9000", "type": "browser"},
    "langfuse": {"url": "http://72.62.237.198:3099", "type": "browser"},
    "uptime kuma": {"url": "http://72.62.237.198:3002/dashboard", "type": "browser"},
    "openclaw": {"url": "http://72.62.237.198:18791", "type": "browser"},
    "litellm": {"url": "http://72.62.237.198:4000", "type": "browser"},
    "qdrant": {"url": "http://72.62.237.198:6333/dashboard", "type": "browser"},
    "nova master": {"url": "http://72.62.237.198:8090", "type": "browser"},
    "novamaster": {"url": "http://72.62.237.198:8090", "type": "browser"},
    "ollama vps": {"url": "http://72.62.237.198:11434", "type": "browser"},
    "jarvis vps": {"url": "http://72.62.237.198:8888", "type": "browser"},
    # Local Services
    "jarvis": {"url": "http://127.0.0.1:8888", "type": "browser"},
    "jarvis cockpit": {"url": "http://127.0.0.1:8888", "type": "browser"},
    "ollama": {"url": "http://127.0.0.1:11434", "type": "browser"},
    "comfyui": {"url": "http://127.0.0.1:8188", "type": "browser"},
    "comfy ui": {"url": "http://127.0.0.1:8188", "type": "browser"},
    "vibevoice": {"url": "http://127.0.0.1:8094", "type": "browser"},
    "clawmem": {"url": "http://127.0.0.1:7438", "type": "browser"},
    "launcher": {"url": "file:///mnt/c/Users/roseo/OneDrive/Bureaublad/alle novamasters%20toools%20en%20launcher%202026/NovaMaster%20Launcher.html", "type": "browser"},
    "dashboard": {"url": "file:///mnt/c/Users/roseo/OneDrive/Bureaublad/alle novamasters%20toools%20en%20launcher%202026/NovaMaster%20Launcher.html", "type": "browser"},
    # Start commands (for services that need starting)
    "jarvis start": {"cmd": "bash -c 'cd /home/faramix/Mark-XXX/ui_web && nohup /usr/bin/python3.12 jarvis_webui.py > /tmp/jarvis-cockpit.log 2>&1 &'", "type": "shell"},
    "comfyui start": {"cmd": "systemctl --user start comfyui.service", "type": "shell"},
    "ollama start": {"cmd": "sudo systemctl start ollama.service", "type": "shell"},
}

# System prompt addition for function calling
COMMAND_SYSTEM_PROMPT = """
You are Nova, a voice assistant that can CONTROL SERVICES. When the user asks to OPEN, START, LAUNCH, or GO TO a service, respond with:
[ACTION:open:SERVICE_NAME]
For example:
- "open AionUI" → [ACTION:open:aion ui]
- "start ComfyUI" → [ACTION:open:comfyui]  
- "open the launcher" → [ACTION:open:launcher]
- "check jarvis" → [ACTION:open:jarvis]
- "open grafana" → [ACTION:open:grafana]
- "go to n8n" → [ACTION:open:n8n]
- "open the dashboard" → [ACTION:open:dashboard]

Available services: aion ui, open webui, hermes, n8n, space agent, grafana, portainer, langfuse, uptime kuma, openclaw, litellm, qdrant, jarvis, comfyui, ollama, vibevoice, clawmem, launcher, dashboard, nova master

If the user just asks a question (no action), answer normally without any [ACTION] tags.
Keep responses SHORT — one or two sentences max for voice.
"""


def find_service(name: str) -> dict:
    """Find a service by name (fuzzy match)."""
    name_lower = name.lower().strip()
    # Exact match
    if name_lower in SERVICES:
        return SERVICES[name_lower]
    # Partial match
    for key, val in SERVICES.items():
        if name_lower in key or key in name_lower:
            return val
    return None


def execute_open(service_name: str) -> str:
    """Open a service URL in the default browser."""
    service = find_service(service_name)
    if not service:
        return f"Service '{service_name}' not found. Say 'launcher' to see all services."
    
    url = service.get("url")
    if url:
        # Use cmd.exe start for WSL to open in Windows browser
        try:
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return f"Opening {service_name}"
        except Exception as e:
            logger.error(f"Failed to open {url}: {e}")
            return f"Failed to open {service_name}: {e}"
    return f"No URL for {service_name}"


def execute_shell(cmd: str) -> str:
    """Execute a shell command."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return f"Done: {result.stdout[:100]}" if result.stdout else "Done"
    except Exception as e:
        return f"Error: {e}"


def process_response(text: str) -> str:
    """
    Process LLM response, execute any [ACTION:xxx] tags, 
    and return clean text for TTS.
    """
    actions = re.findall(r'\[ACTION:(\w+):([^\]]+)\]', text)
    
    responses = []
    for action_type, target in actions:
        if action_type == "open":
            result = execute_open(target)
            responses.append(result)
    
    # Remove action tags from text for TTS
    clean_text = re.sub(r'\[ACTION:[^\]]+\]', '', text).strip()
    
    # If we had actions, append their results
    if responses:
        action_text = ". ".join(responses)
        if clean_text:
            return f"{clean_text}. {action_text}"
        return action_text
    
    return clean_text if clean_text else text


def get_enhanced_system_prompt() -> str:
    """Return the enhanced system prompt with command capabilities."""
    return COMMAND_SYSTEM_PROMPT
