# `llm-cli`

A simple command-line tool that uses AI (Claude, OpenAI, or Ollama) to suggest shell commands, generate code snippets, or explain programming concepts based on natural language descriptions.

## Features

- **Command suggestions**: Get shell commands from natural language descriptions
- **Code gen**: Generate code snippets with the `--code` flag
- **Explanations**: Get brief explanations of commands/concepts with the `--explain` flag
- **Multi-API support**: Works with Anthropic Claude, OpenAI GPT models, local Ollama models, and any OpenAI-compatible server

## Installation

Install Go, run `make install`.

## Setup

For Claude, the preferred setup is a logged-in session via the
[Anthropic CLI](https://github.com/anthropics/anthropic-cli) -- no static key
to manage:

```bash
ant auth login
```

Or set one of the following environment variables:

```bash
export ANTHROPIC_API_KEY=your_claude_api_key
export OPENAI_API_KEY=your_openai_api_key
export LLM_MODEL=your_local_model_name
export OLLAMA_MODEL=your_ollama_model_name
```

The tool will automatically use whichever credential or model is available.
Priority order: Claude > OpenAI > Local > Ollama.

Claude credentials resolve in this order: logged-in `ant` profile >
`ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_API_KEY`. When a logged-in profile exists,
`llm` fetches a short-lived OAuth token from it on each run (via
`ant auth print-credentials`), so it keeps working across token refreshes and
ignores any stale exported API key.

### Local / OpenAI-compatible servers

The `LLM_*` provider works with any OpenAI-compatible server and defaults to
`rainman.j.co` (`http://rainman.j.co:30000/v1`, model `Qwen3.8-27B`).
Set `LLM_MODEL` alone to use the defaults, or override:

```bash
export LLM_BASE_URL=http://your-server:30000/v1   # OpenAI-compatible base URL
export LLM_MODEL=your-model-name
export LLM_API_KEY=your_key                       # optional
```

The Claude model can be pinned with:

```bash
export CLAUDE_MODEL=claude-sonnet-5-5             # default
```

## Usage

### Basic Commands
```bash
% llm search for files larger than 100MB
find . -type f -size +100M

% llm decrypt with gpg, unzip, filter for files larger than 10gb, sum the third column
gpg --decrypt archive.gpg | unzip -p - | find . -type f -size +10G -exec awk '{sum += $3} END {print sum}' {} +
```

### Code Generation
```bash
% llm -c python to port scan 10.8.1.1/24
import socket
from concurrent.futures import ThreadPoolExecutor

def scan_port(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((ip, port))
        sock.close()
        if result == 0:
            print(f"{ip}:{port} open")
    except:
        pass

def scan_host(host):
    ip = f"10.8.1.{host}"
    with ThreadPoolExecutor(max_workers=100) as executor:
        for port in range(1, 1025):
            executor.submit(scan_port, ip, port)

with ThreadPoolExecutor(max_workers=50) as executor:
    for i in range(1, 255):
        executor.submit(scan_host, i)
```

### Explanations
```bash
% llm --explain what does grep -r do
grep -r performs a recursive search through directories...

% llm -x explain the find command
The find command searches for files and directories...
```

## Options

- `-b, --backend`: Force a specific backend (`claude`, `openai`, `local`, or `ollama`) instead of auto-detecting
- `--local`: Use the local backend (shorthand for `-b local`)
- `-c, --code`: Code generation mode
- `-x, --explain`: Explanation mode  
- `-h, --help`: Show help message
- `-v, --version`: Show version

```bash
% llm -b local "list files by size"    # force rainman.j.co even if ANTHROPIC_API_KEY is set
```

## Models Used

- **Claude**: `claude-sonnet-5-5` (override with `CLAUDE_MODEL`)
- **OpenAI**: `gpt-4o-mini`
- **Local**: Any OpenAI-compatible server; defaults to `Qwen3.8-27B` on `rainman.j.co`
- **Ollama**: Any locally installed model (e.g., llama2, mistral, codellama)

## License

MIT
