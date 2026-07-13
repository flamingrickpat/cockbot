#!/usr/bin/env python3
"""
Minimal local coding agent for Windows.

Features:
- OpenAI-compatible Chat Completions endpoint, including llama.cpp server.
- One unrestricted PowerShell tool for filesystem work and command execution.
- MCP servers loaded from a JSON config discovered from the working directory.
- Pi/Codex-compatible SKILL.md discovery with progressive disclosure.
- AGENTS.md and CLAUDE.md project instructions.
- In-memory follow-up conversation only; no session files or persistence.

Install:
    pip install "pydantic-ai-slim[openai,mcp]>=2,<3" pyyaml

Example:
    python local_coding_agent.py --model qwen --base-url http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import yaml
from pydantic_ai import Agent, Tool
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider


APP_NAME = "Local Coding Agent"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_POWERSHELL_TIMEOUT = 180
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 60_000
DEFAULT_MAX_SKILL_INDEX_CHARS = 16_000
DEFAULT_MAX_PROJECT_INSTRUCTIONS_CHARS = 24_000

MCP_CONFIG_CANDIDATES = (
    ".mcp.json",
    "mcp.json",
    "mcp_config.json",
    ".pi/mcp.json",
    ".codex/mcp.json",
)

FRONTMATTER_RE = re.compile(
    r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class Settings:
    cwd: Path
    base_url: str
    api_key: str
    model: str
    mcp_config: Path | None
    powershell_exe: str
    powershell_timeout: int
    max_tool_output_chars: int
    max_skill_index_chars: int
    max_project_instructions_chars: int
    temperature: float


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    skill_file: Path
    priority: int


@dataclass(frozen=True)
class DiscoveryResult:
    skills: list[Skill]
    skill_errors: list[str]
    instruction_files: list[Path]
    project_instructions: str
    mcp_server_names: list[str]


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A minimal local coding agent with unrestricted PowerShell, "
            "MCP tools, skills, and in-memory follow-up conversation."
        )
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "--base-url",
        default=env_first("LOCAL_AGENT_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        help=f"OpenAI-compatible /v1 base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--api-key",
        default=env_first("LOCAL_AGENT_API_KEY", "OPENAI_API_KEY") or "local",
        help="API key or placeholder accepted by the local endpoint.",
    )
    parser.add_argument(
        "--model",
        default=env_first("LOCAL_AGENT_MODEL", "OPENAI_MODEL"),
        help="Model ID/llama.cpp alias. If omitted, /v1/models is queried.",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "Explicit MCP JSON config. Otherwise searches the working directory "
            "and its parents for .mcp.json, mcp.json, or mcp_config.json."
        ),
    )
    parser.add_argument(
        "--powershell-timeout",
        type=int,
        default=int(os.environ.get("LOCAL_AGENT_POWERSHELL_TIMEOUT", DEFAULT_POWERSHELL_TIMEOUT)),
        help=(
            "Default PowerShell timeout in seconds. The model can pass 0 for no timeout. "
            f"Default: {DEFAULT_POWERSHELL_TIMEOUT}."
        ),
    )
    parser.add_argument(
        "--max-tool-output",
        type=int,
        default=int(os.environ.get("LOCAL_AGENT_MAX_TOOL_OUTPUT", DEFAULT_MAX_TOOL_OUTPUT_CHARS)),
        help="Maximum PowerShell result characters returned to the model.",
    )
    parser.add_argument(
        "--max-skill-index",
        type=int,
        default=int(os.environ.get("LOCAL_AGENT_MAX_SKILL_INDEX", DEFAULT_MAX_SKILL_INDEX_CHARS)),
        help="Maximum characters used for the initial skill metadata index.",
    )
    parser.add_argument(
        "--max-project-instructions",
        type=int,
        default=int(
            os.environ.get(
                "LOCAL_AGENT_MAX_PROJECT_INSTRUCTIONS",
                DEFAULT_MAX_PROJECT_INSTRUCTIONS_CHARS,
            )
        ),
        help="Maximum characters loaded from AGENTS.md and CLAUDE.md files.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LOCAL_AGENT_TEMPERATURE", "0.1")),
        help="Model sampling temperature. Default: 0.1.",
    )
    return parser.parse_args()


def discover_model_id(base_url: str, api_key: str) -> str | None:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    models = payload.get("data", [])
    for model in models:
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None


def find_powershell() -> str:
    for executable in ("pwsh.exe", "pwsh", "powershell.exe", "powershell"):
        resolved = shutil.which(executable)
        if resolved:
            return resolved
    raise RuntimeError(
        "PowerShell was not found. Install PowerShell 7 (pwsh.exe) or run on Windows "
        "with powershell.exe available on PATH."
    )


def ancestor_directories_nearest_first(cwd: Path) -> list[Path]:
    directories: list[Path] = []
    current = cwd.resolve()
    while True:
        directories.append(current)
        if current.parent == current:
            break
        current = current.parent
    return directories


def find_mcp_config(cwd: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCP config does not exist: {path}")
        return path

    for directory in ancestor_directories_nearest_first(cwd):
        for relative_name in MCP_CONFIG_CANDIDATES:
            candidate = directory / relative_name
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_mcp_server_names(config_path: Path | None) -> list[str]:
    if config_path is None:
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError(
            f"MCP config must contain a top-level 'mcpServers' object: {config_path}"
        )
    return [str(name) for name in servers]


def skill_roots(cwd: Path) -> list[Path]:
    roots: list[Path] = []

    # Nearest project-local skills take precedence over parent/global skills.
    for directory in ancestor_directories_nearest_first(cwd):
        roots.extend(
            [
                directory / ".agents" / "skills",
                directory / ".pi" / "skills",
                directory / ".codex" / "skills",
            ]
        )

    roots.extend(
        [
            Path.home() / ".agents" / "skills",
            Path.home() / ".pi" / "agent" / "skills",
            Path.home() / ".codex" / "skills",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def fallback_skill_description(body: str, skill_name: str) -> str:
    paragraphs = re.split(r"\r?\n\s*\r?\n", body)
    for paragraph in paragraphs:
        lines = []
        for raw_line in paragraph.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            if line.startswith("#"):
                line = line.lstrip("#").strip()
                if line.casefold() == skill_name.casefold():
                    continue
            lines.append(line)
        if lines:
            return " ".join(lines)
    return f"Reusable workflow from {skill_name}."


def parse_skill(skill_file: Path, priority: int) -> Skill:
    text = skill_file.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    metadata: dict[str, Any] = {}
    body = text

    if match:
        loaded = yaml.safe_load(match.group(1))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("YAML frontmatter must be an object")
        metadata = loaded or {}
        body = text[match.end() :]

    name = str(metadata.get("name") or skill_file.parent.name).strip()
    description_value = metadata.get("description")
    description = (
        str(description_value).strip()
        if description_value is not None
        else fallback_skill_description(body, name)
    )
    description = re.sub(r"\s+", " ", description).strip()

    if not name:
        raise ValueError("Skill name is empty")
    if not description:
        raise ValueError("Skill description is empty")

    return Skill(
        name=name,
        description=description[:800],
        skill_file=skill_file.resolve(),
        priority=priority,
    )


def load_skills(cwd: Path) -> tuple[list[Skill], list[str]]:
    skills_by_name: dict[str, Skill] = {}
    seen_files: set[Path] = set()
    errors: list[str] = []

    for priority, root in enumerate(skill_roots(cwd)):
        if not root.is_dir():
            continue
        for skill_file in sorted(root.rglob("SKILL.md")):
            resolved = skill_file.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            try:
                skill = parse_skill(resolved, priority)
            except Exception as exc:
                errors.append(f"{resolved}: {exc}")
                continue
            skills_by_name.setdefault(skill.name.casefold(), skill)

    skills = sorted(
        skills_by_name.values(),
        key=lambda skill: (skill.priority, skill.name.casefold()),
    )
    return skills, errors


def format_skill_index(skills: list[Skill], max_chars: int) -> str:
    if not skills:
        return "<available_skills />"

    opening = "<available_skills>\n"
    closing = "</available_skills>"
    chunks = [opening]
    current_length = len(opening) + len(closing)
    omitted = 0

    for skill in skills:
        chunk = (
            "  <skill>\n"
            f"    <name>{escape(skill.name)}</name>\n"
            f"    <description>{escape(skill.description)}</description>\n"
            f"    <location>{escape(str(skill.skill_file))}</location>\n"
            "  </skill>\n"
        )
        if current_length + len(chunk) > max_chars:
            omitted += 1
            continue
        chunks.append(chunk)
        current_length += len(chunk)

    if omitted:
        chunks.append(
            f"  <omitted count={quoteattr(str(omitted))}>"
            "Additional skills exist but were omitted from the initial context budget. "
            "Search the documented skill roots with PowerShell if none of the listed skills applies."
            "</omitted>\n"
        )
    chunks.append(closing)
    return "".join(chunks)


def discover_instruction_files(cwd: Path) -> list[Path]:
    files: list[Path] = []

    global_agents = Path.home() / ".pi" / "agent" / "AGENTS.md"
    if global_agents.is_file():
        files.append(global_agents.resolve())

    # General instructions first, nearest/local instructions last.
    for directory in reversed(ancestor_directories_nearest_first(cwd)):
        for filename in ("AGENTS.md", "CLAUDE.md"):
            candidate = directory / filename
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in files:
                    files.append(resolved)
    return files


def load_project_instructions(files: list[Path], max_chars: int) -> str:
    if not files:
        return "<project_instructions />"

    opening = "<project_instructions>\n"
    closing = "</project_instructions>"
    chunks = [opening]
    current_length = len(opening) + len(closing)
    omitted = 0

    for path in files:
        content = path.read_text(encoding="utf-8")
        chunk = (
            f"  <instruction_file path={quoteattr(str(path))}>\n"
            f"{content}\n"
            "  </instruction_file>\n"
        )
        if current_length + len(chunk) > max_chars:
            omitted += 1
            continue
        chunks.append(chunk)
        current_length += len(chunk)

    if omitted:
        chunks.append(
            f"  <omitted count={quoteattr(str(omitted))}>"
            "Some instruction files exceeded the initial context budget. Read them with PowerShell when relevant."
            "</omitted>\n"
        )
    chunks.append(closing)
    return "".join(chunks)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n\n... [{len(text) - max_chars:,} characters omitted] ...\n\n"
    remaining = max_chars - len(marker)
    head = max(0, remaining // 2)
    tail = max(0, remaining - head)
    return text[:head] + marker + text[-tail:]


def make_powershell_tool(settings: Settings) -> Tool[Any]:
    default_timeout = settings.powershell_timeout

    def powershell(command: str, timeout_seconds: int = default_timeout) -> str:
        """Execute an arbitrary PowerShell script with the full permissions of this agent process.

        Use this for all filesystem inspection and editing, Git, Python, Node.js, TypeScript,
        package managers, tests, processes, network commands, registry access, and executable calls.
        Each call starts a fresh PowerShell process in the agent working directory, so state such as
        Set-Location and variables does not persist between calls. Combine dependent commands in one
        script or use absolute paths. A timeout_seconds value of 0 disables the timeout.

        Args:
            command: Complete PowerShell script to execute. Multiline scripts are supported.
            timeout_seconds: Maximum execution time in seconds, or 0 for no timeout.
        """

        print("\n[powershell]", flush=True)
        print(command.rstrip(), flush=True)

        preamble = (
            "$OutputEncoding = [Console]::OutputEncoding = "
            "[System.Text.UTF8Encoding]::new($false)\n"
            "$ProgressPreference = 'SilentlyContinue'\n"
        )
        encoded = base64.b64encode((preamble + command).encode("utf-16-le")).decode("ascii")
        timeout = None if timeout_seconds <= 0 else timeout_seconds

        try:
            completed = subprocess.run(
                [
                    settings.powershell_exe,
                    "-NoLogo",
                    "-NoProfile",
                    "-EncodedCommand",
                    encoded,
                ],
                cwd=settings.cwd,
                env=os.environ.copy(),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace")
            stderr = completed.stderr.decode("utf-8", errors="replace")
            result = (
                f"exit_code: {completed.returncode}\n"
                f"stdout:\n{stdout if stdout else '(empty)'}\n"
                f"stderr:\n{stderr if stderr else '(empty)'}"
            )
            print(f"[powershell exit {completed.returncode}]", flush=True)
            return truncate_middle(result, settings.max_tool_output_chars)
        except subprocess.TimeoutExpired as exc:
            stdout_bytes = exc.stdout or b""
            stderr_bytes = exc.stderr or b""
            if isinstance(stdout_bytes, str):
                stdout = stdout_bytes
            else:
                stdout = stdout_bytes.decode("utf-8", errors="replace")
            if isinstance(stderr_bytes, str):
                stderr = stderr_bytes
            else:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
            result = (
                f"timed_out: true\n"
                f"timeout_seconds: {timeout_seconds}\n"
                f"stdout_before_timeout:\n{stdout if stdout else '(empty)'}\n"
                f"stderr_before_timeout:\n{stderr if stderr else '(empty)'}"
            )
            print(f"[powershell timed out after {timeout_seconds}s]", flush=True)
            return truncate_middle(result, settings.max_tool_output_chars)

    return Tool(
        powershell,
        takes_ctx=False,
        name="powershell",
        sequential=True,
        max_retries=1,
        strict=False,
    )


def build_system_prompt(
    settings: Settings,
    skill_index: str,
    project_instructions: str,
    mcp_server_names: list[str],
) -> str:
    mcp_summary = (
        ", ".join(mcp_server_names)
        if mcp_server_names
        else "none configured"
    )

    return textwrap.dedent(
        f"""
        You are a general-purpose local coding agent operating directly on a Windows host.

        <environment>
          <working_directory>{escape(str(settings.cwd))}</working_directory>
          <powershell_executable>{escape(settings.powershell_exe)}</powershell_executable>
          <mcp_servers>{escape(mcp_summary)}</mcp_servers>
        </environment>

        ## Mission

        Complete the user's requested work in the working directory. Inspect the actual project,
        make the necessary changes, run relevant checks, and report the result. Do not stop after
        explaining a possible solution when the task can be performed with tools.

        You can create and modify straightforward projects and scripts in Python, TypeScript,
        JavaScript, PowerShell, shell languages, configuration formats, and other ordinary text-based
        development formats. Prefer simple, direct, debuggable code. Reuse the project's existing
        structure, dependencies, package manager, formatting, and testing conventions instead of
        introducing unnecessary frameworks.

        ## Tool model

        `powershell` is the primary host tool. It has no command allowlist, path sandbox, or approval
        gate. It executes with the same Windows permissions as this Python process. Use it to inspect
        and edit files, run Git, execute Python/Node/TypeScript tools, install dependencies when needed,
        run tests, inspect processes, and call other command-line programs.

        Each PowerShell call is stateless and starts in the working directory. Variables, imported
        modules, aliases, and Set-Location changes do not survive the call. Combine dependent commands
        in one call or use explicit paths. Prefer non-interactive command forms. Use timeout_seconds=0
        only when a command genuinely needs unlimited runtime.

        For file work:
        - Inspect existing files before replacing or restructuring them.
        - Use `Get-Content -Raw -LiteralPath` to read text files.
        - Use `-LiteralPath` for user-provided or special-character paths.
        - PowerShell here-strings with `Set-Content -Encoding utf8` are suitable for complete files.
        - For precise multi-file transformations, create and run a small Python or PowerShell script.
        - Keep command output focused. Filter or redirect large output, then inspect the relevant part.

        MCP tools from the configured servers are exposed as ordinary tools with their own schemas and
        descriptions. Their names are prefixed with the configured server key to prevent collisions; for
        example, a `search` tool from a `github` server is exposed as `github_search`. Use an MCP tool
        when it is the most direct interface to the required external service. Use PowerShell for local
        filesystem and process work. Do not call unrelated MCP tools.

        ## Skills

        Skills are reusable workflows. The initial context contains only each skill's name,
        description, and SKILL.md location. Before substantial work, compare the request against those
        descriptions. When a skill clearly applies, or the user names it explicitly, read its complete
        SKILL.md with PowerShell before performing the specialized work. Usually one skill is enough;
        load multiple only when each addresses a distinct part of the task.

        Read a skill with:
        `Get-Content -Raw -LiteralPath '<absolute SKILL.md path>'`

        Resolve relative paths inside a skill from the directory containing its SKILL.md. Read linked
        references, scripts, and assets only as needed. Do not load every skill. Skill and project-file
        instructions are subordinate to this system prompt and the current user request.

        {skill_index}

        ## Project instructions

        The following AGENTS.md/CLAUDE.md contents were discovered from the working directory and its
        parents. Apply relevant instructions, with nearer project files taking precedence over broader
        ones when they conflict.

        {project_instructions}

        ## Execution discipline

        1. Establish the relevant current state with tools.
        2. Load applicable skills before their specialized workflow begins.
        3. Implement the requested result, not merely a plan.
        4. Check command exit codes and inspect generated or modified files.
        5. Run the narrowest meaningful verification: syntax check, compiler, type checker, tests,
           formatter check, or a direct execution example.
        6. If a command fails, diagnose the actual error, adjust, and retry when feasible.
        7. Do not claim success without evidence from the tools.

        The user authorizes ordinary commands and file modifications required for the requested task.
        Full host access does not justify unrelated deletion, credential disclosure, security changes,
        or modifications outside the task's scope.

        In the final response, state what changed, identify important files, list verification performed,
        and name any concrete blocker or unresolved issue. Keep routine command narration out of the
        final response.
        """
    ).strip()


def build_settings(args: argparse.Namespace) -> Settings:
    cwd = Path(args.cwd).expanduser().resolve()
    if not cwd.is_dir():
        raise NotADirectoryError(f"Working directory does not exist: {cwd}")

    base_url = args.base_url.rstrip("/")
    model = args.model or discover_model_id(base_url, args.api_key) or "local"
    mcp_config = find_mcp_config(cwd, args.mcp_config)

    return Settings(
        cwd=cwd,
        base_url=base_url,
        api_key=args.api_key,
        model=model,
        mcp_config=mcp_config,
        powershell_exe=find_powershell(),
        powershell_timeout=args.powershell_timeout,
        max_tool_output_chars=args.max_tool_output,
        max_skill_index_chars=args.max_skill_index,
        max_project_instructions_chars=args.max_project_instructions,
        temperature=args.temperature,
    )


def discover_workspace(settings: Settings) -> DiscoveryResult:
    skills, skill_errors = load_skills(settings.cwd)
    instruction_files = discover_instruction_files(settings.cwd)
    project_instructions = load_project_instructions(
        instruction_files,
        settings.max_project_instructions_chars,
    )
    mcp_server_names = read_mcp_server_names(settings.mcp_config)
    return DiscoveryResult(
        skills=skills,
        skill_errors=skill_errors,
        instruction_files=instruction_files,
        project_instructions=project_instructions,
        mcp_server_names=mcp_server_names,
    )


def build_agent(settings: Settings, discovery: DiscoveryResult) -> Agent[Any, str]:
    skill_index = format_skill_index(
        discovery.skills,
        settings.max_skill_index_chars,
    )
    system_prompt = build_system_prompt(
        settings,
        skill_index,
        discovery.project_instructions,
        discovery.mcp_server_names,
    )

    profile = OpenAIModelProfile(
        openai_supports_strict_tool_definition=False,
        openai_chat_supports_multiple_system_messages=False,
    )
    model = OpenAIChatModel(
        settings.model,
        provider=OpenAIProvider(
            base_url=settings.base_url,
            api_key=settings.api_key,
        ),
        profile=profile,
    )

    toolsets = (
        load_mcp_toolsets(settings.mcp_config)
        if settings.mcp_config is not None
        else []
    )

    return Agent(
        model=model,
        instructions=system_prompt,
        tools=[make_powershell_tool(settings)],
        toolsets=toolsets,
        model_settings={"temperature": settings.temperature},
        retries=3,
        max_concurrency=1,
    )


def print_startup(settings: Settings, discovery: DiscoveryResult) -> None:
    print(f"\n{APP_NAME}")
    print(f"  cwd:        {settings.cwd}")
    print(f"  endpoint:   {settings.base_url}")
    print(f"  model:      {settings.model}")
    print(f"  PowerShell: {settings.powershell_exe}")
    print(f"  skills:     {len(discovery.skills)}")
    print(f"  instructions: {len(discovery.instruction_files)} file(s)")
    if settings.mcp_config:
        names = ", ".join(discovery.mcp_server_names) or "no servers"
        print(f"  MCP config: {settings.mcp_config}")
        print(f"  MCP servers: {names}")
    else:
        print("  MCP config: none found")

    for error in discovery.skill_errors:
        print(f"  skill warning: {error}")

    print("\nCommands: /quit, /exit, /clear, /info, /paste")
    print("Use /paste for multiline input; finish with a line containing only /end.\n")


def read_user_prompt() -> str | None:
    try:
        first_line = input("You> ")
    except EOFError:
        return None

    if first_line.strip().casefold() != "/paste":
        return first_line

    print("Paste mode. Enter /end on its own line to submit.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().casefold() == "/end":
            break
        lines.append(line)
    return "\n".join(lines)


async def interactive_loop(
    agent: Agent[Any, str],
    settings: Settings,
    discovery: DiscoveryResult,
) -> None:
    message_history = None

    async with agent:
        while True:
            prompt = read_user_prompt()
            if prompt is None:
                print()
                return

            command = prompt.strip().casefold()
            if command in {"/quit", "/exit", "quit", "exit"}:
                return
            if command == "/clear":
                message_history = None
                print("In-memory conversation cleared.\n")
                continue
            if command == "/info":
                print_startup(settings, discovery)
                continue
            if not prompt.strip():
                continue

            try:
                result = await agent.run(
                    prompt,
                    message_history=message_history,
                )
            except Exception as exc:
                print(f"\n[agent error] {type(exc).__name__}: {exc}\n")
                continue

            message_history = result.all_messages()
            print(f"\nAgent> {result.output}\n")


async def async_main() -> None:
    args = parse_args()
    settings = build_settings(args)

    # Relative MCP commands and all agent subprocesses should share the selected workspace.
    os.chdir(settings.cwd)

    discovery = discover_workspace(settings)
    agent = build_agent(settings, discovery)
    print_startup(settings, discovery)
    await interactive_loop(agent, settings, discovery)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"{APP_NAME} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
