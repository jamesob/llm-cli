# `llm-cli`

A simple command-line tool that uses AI (Claude, OpenAI, or Ollama) to suggest shell commands, generate code snippets, or explain programming concepts based on natural language descriptions.

## Features

- **Command suggestions**: Get shell commands from natural language descriptions
- **Code gen**: Generate code snippets with the `--code` flag
- **Explanations**: Get brief explanations of commands/concepts with the `--explain` flag
- **Multi-API support**: Works with Anthropic Claude, OpenAI GPT models, and local Ollama models

## Installation

Install Go, run `make install`.

## Setup

Set one of the following environment variables:

```bash
export ANTHROPIC_API_KEY=your_claude_api_key
export OPENAI_API_KEY=your_openai_api_key
export OLLAMA_MODEL=your_ollama_model_name
```

The tool will automatically use whichever key or model is available (Claude takes priority if multiple are set).

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

- `-c, --code`: Code generation mode
- `-x, --explain`: Explanation mode  
- `-h, --help`: Show help message
- `-v, --version`: Show version

## Models Used

- **Claude**: `claude-sonnet-4-20250514`
- **OpenAI**: `gpt-4o-mini`
- **Ollama**: Any locally installed model (e.g., llama2, mistral, codellama)

## License

MIT
