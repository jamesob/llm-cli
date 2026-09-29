package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"regexp"
	"time"
)

const (
	claudeAPIURL       = "https://api.anthropic.com/v1/messages"
	openaiAPIURL       = "https://api.openai.com/v1/chat/completions"
	ollamaAPIURL       = "http://localhost:11434/api/generate"
	defaultClaudeModel = "claude-sonnet-5-5"
	defaultLocalURL    = "http://rainman.j.co:30000/v1"
	defaultLocalModel  = "Qwen3.8-27B"
	version            = "1.0.0"
)

// Claude API structs
type ClaudeRequest struct {
	Model     string    `json:"model"`
	MaxTokens int       `json:"max_tokens"`
	Messages  []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ClaudeResponse struct {
	Content []ContentBlock `json:"content"`
	Error   *APIError      `json:"error,omitempty"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// OpenAI API structs
type OpenAIRequest struct {
	Model       string           `json:"model"`
	Messages    []OpenAIMessage  `json:"messages"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIResponse struct {
	Choices []OpenAIChoice `json:"choices"`
	Error   *APIError      `json:"error,omitempty"`
}

type OpenAIChoice struct {
	Message OpenAIMessage `json:"message"`
}

// Ollama API structs
type OllamaRequest struct {
	Model    string `json:"model"`
	Prompt   string `json:"prompt"`
	Stream   bool   `json:"stream"`
}

type OllamaResponse struct {
	Response string    `json:"response"`
	Error    *APIError `json:"error,omitempty"`
}

// Common error struct
type APIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type APIProvider int

const (
	Claude APIProvider = iota
	OpenAI
	Local
	Ollama
)

// How to authenticate against the Claude API.
type claudeAuthKind int

const (
	authAPIKey claudeAuthKind = iota // x-api-key header
	authBearer                       // ANTHROPIC_AUTH_TOKEN: Authorization: Bearer
	authOAuth                        // ant profile token: Bearer + oauth beta header
)

type providerConfig struct {
	provider   APIProvider
	apiKey     string
	model      string
	baseURL    string // Local provider only
	claudeAuth claudeAuthKind
}

func main() {
	start := time.Now()

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	// Define flags
	var codeMode bool
	var explainMode bool
	var backend string
	var localMode bool

	// Custom flag set to handle both short and long flags
	flagSet := flag.NewFlagSet("llm", flag.ExitOnError)
	flagSet.BoolVar(&codeMode, "code", false, "Code generation mode")
	flagSet.BoolVar(&codeMode, "c", false, "Code generation mode (short)")
	flagSet.BoolVar(&explainMode, "explain", false, "Explanation mode")
	flagSet.BoolVar(&explainMode, "x", false, "Explanation mode (short)")
	flagSet.StringVar(&backend, "backend", "", "Force a specific backend: claude, openai, local, or ollama")
	flagSet.StringVar(&backend, "b", "", "Force a specific backend (short)")
	flagSet.BoolVar(&localMode, "local", false, "Use the local backend (shorthand for -b local)")

	// Custom usage function
	flagSet.Usage = printUsage

	// Handle help and version flags
	if os.Args[1] == "--help" || os.Args[1] == "-h" {
		printUsage()
		return
	}
	if os.Args[1] == "--version" || os.Args[1] == "-v" {
		fmt.Printf("llm version %s\n", version)
		return
	}

	// Parse flags and get remaining arguments
	if err := flagSet.Parse(os.Args[1:]); err != nil {
		os.Exit(1)
	}

	query := strings.Join(flagSet.Args(), " ")

	if localMode && backend != "" {
		fmt.Fprintf(os.Stderr, "Error: cannot use both --local and -b/--backend\n")
		os.Exit(1)
	}
	if localMode {
		backend = "local"
	}

	// Determine which API to use
	cfg, err := determineProvider(backend)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		if backend == "" {
			fmt.Fprintf(os.Stderr, "Log in with `ant auth login`, or set one of the following environment variables:\n")
			fmt.Fprintf(os.Stderr, "  export ANTHROPIC_API_KEY=your_claude_api_key\n")
			fmt.Fprintf(os.Stderr, "  export OPENAI_API_KEY=your_openai_api_key\n")
			fmt.Fprintf(os.Stderr, "  export LLM_MODEL=your_local_model_name\n")
			fmt.Fprintf(os.Stderr, "  export OLLAMA_MODEL=your_ollama_model_name\n")
		}
		os.Exit(1)
	}

	// Get system context
	osInfo := runtime.GOOS
	shell := getShell()
	prompt := ""
	renderAsMd := false

	if (codeMode) {
		prompt = fmt.Sprintf(`You are a code-writing assistant. The user is on %s using %s shell and needs a code snippet.

User request: %s

Respond with ONLY the code that would accomplish this task. Do not include explanations, code comments, markdown formatting, or extra text. Write the most concise code possible, and prefer use of standard libraries to third parties.
`, osInfo, shell, query)

	} else if (explainMode) {
		prompt = fmt.Sprintf(`You are a programming expert. The user is on %s using %s shell and needs a brief explanation of a CLI command or a programming library or concept.

User request: %s

Respond with ONLY a very brief, concise description of the concept or solution. The answer should not exceed 2 paragraphs.
`, osInfo, shell, query)
		renderAsMd = true

	} else {
		prompt = fmt.Sprintf(`You are a command-line assistant. The user is on %s using %s shell and needs a command suggestion.

User request: %s

Respond with ONLY the command(s) that would accomplish this task. Do not include explanations, markdown formatting, or extra text. If multiple commands are needed, put each on a separate line.

Examples:
- For "search for foo in directory" → "grep -R foo ."
- For "list files by size" → "ls -laSh"
- For "find large files" → "find . -type f -size +100M"`, osInfo, shell, query)
		renderAsMd = true
	}

	var response string
	switch cfg.provider {
	case Claude:
		response, err = queryClaudeAPI(cfg, prompt)
	case OpenAI:
		response, err = queryOpenAIAPI(cfg.apiKey, prompt)
	case Local:
		response, err = queryLocalAPI(cfg, prompt)
	case Ollama:
		response, err = queryOllamaAPI(cfg.model, prompt)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if renderAsMd {
		fmt.Println(RenderMarkdown(response))
	} else {
		fmt.Println(response)
	}

	elapsed := time.Since(start)
	if elapsed > 5*time.Second {
		fmt.Fprintf(os.Stderr, "%s%.2fs%s\n", Dim, elapsed.Seconds(), Reset)
	}
}

func printUsage() {
	fmt.Printf(`llm - Multi-API Command Suggester v%s

USAGE:
    llm <description of what you want to do>

EXAMPLES:
    llm search for foo in directory
    llm list files by size
    llm find files modified today
    llm compress this directory
    llm show disk usage
	llm --code write a python function to diff a file
	llm --explain explain the cp command

SETUP:
    Log in with the Anthropic CLI (preferred; no static key to manage):
    ant auth login

    Or set one of the following environment variables:
    export ANTHROPIC_API_KEY=your_claude_api_key
    export OPENAI_API_KEY=your_openai_api_key
    export LLM_MODEL=your_local_model_name
    export OLLAMA_MODEL=your_ollama_model_name

    The script will automatically detect which credential or model is available and use the corresponding service.
    Priority order: Claude > OpenAI > Local > Ollama
    Claude credentials resolve as: logged-in ant profile > ANTHROPIC_AUTH_TOKEN > ANTHROPIC_API_KEY

    The Local provider talks to any OpenAI-compatible server (defaults to
    rainman.j.co). Optional overrides:
    export LLM_BASE_URL=http://rainman.j.co:30000/v1
    export LLM_API_KEY=your_key
    export CLAUDE_MODEL=claude-sonnet-5-5

OPTIONS:
    -b, --backend  Force a specific backend: claude, openai, local, or ollama
    --local        Use the local backend (shorthand for -b local)
    -h, --help     Show this help message
    -v, --version  Show version information
    -c, --code     Code generation mode
    -x, --explain  Explanation mode
 `, version)
}

func getShell() string {
	shell := os.Getenv("SHELL")
	if shell == "" {
		if runtime.GOOS == "windows" {
			return "cmd/powershell"
		}
		return "sh"
	}
	// Extract just the shell name (e.g., "/bin/bash" -> "bash")
	parts := strings.Split(shell, "/")
	return parts[len(parts)-1]
}

func determineProvider(backend string) (providerConfig, error) {
	// Explicit backend selection
	if backend != "" {
		return backendProvider(backend)
	}

	// Auto-detect in priority order: Claude > OpenAI > Local > Ollama.
	// Claude is available via a logged-in ant profile or env credentials.
	if cfg, err := backendProvider("claude"); err == nil {
		return cfg, nil
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		return backendProvider("openai")
	}
	if os.Getenv("LLM_BASE_URL") != "" || os.Getenv("LLM_MODEL") != "" {
		return backendProvider("local")
	}
	if os.Getenv("OLLAMA_MODEL") != "" {
		return backendProvider("ollama")
	}

	return providerConfig{}, fmt.Errorf("no API key or model found")
}

func backendProvider(backend string) (providerConfig, error) {
	switch backend {
	case "claude":
		model := os.Getenv("CLAUDE_MODEL")
		if model == "" {
			model = defaultClaudeModel
		}
		// Prefer a logged-in session (ant auth login) over static keys.
		if token := claudeSessionToken(); token != "" {
			return providerConfig{provider: Claude, apiKey: token, model: model,
				claudeAuth: authOAuth}, nil
		}
		if token := os.Getenv("ANTHROPIC_AUTH_TOKEN"); token != "" {
			return providerConfig{provider: Claude, apiKey: token, model: model,
				claudeAuth: authBearer}, nil
		}
		if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
			return providerConfig{provider: Claude, apiKey: apiKey, model: model}, nil
		}
		return providerConfig{}, fmt.Errorf(
			"claude backend requires a logged-in session (ant auth login), " +
				"ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY")
	case "openai":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return providerConfig{}, fmt.Errorf("openai backend requires OPENAI_API_KEY")
		}
		return providerConfig{provider: OpenAI, apiKey: apiKey, model: "gpt-4o-mini"}, nil
	case "local", "rainman":
		baseURL := os.Getenv("LLM_BASE_URL")
		if baseURL == "" {
			baseURL = defaultLocalURL
		}
		model := os.Getenv("LLM_MODEL")
		if model == "" {
			model = defaultLocalModel
		}
		return providerConfig{provider: Local, apiKey: os.Getenv("LLM_API_KEY"), model: model, baseURL: baseURL}, nil
	case "ollama":
		model := os.Getenv("OLLAMA_MODEL")
		if model == "" {
			return providerConfig{}, fmt.Errorf("ollama backend requires OLLAMA_MODEL")
		}
		return providerConfig{provider: Ollama, model: model}, nil
	}
	return providerConfig{}, fmt.Errorf("unknown backend %q (expected claude, openai, local, or ollama)", backend)
}

// claudeSessionToken returns a short-lived OAuth access token from a logged-in
// `ant auth login` profile, or "" when the ant CLI or an active profile isn't
// available. print-credentials refreshes an expired token itself, so calling
// it once per invocation is sufficient.
func claudeSessionToken() string {
	antPath, err := exec.LookPath("ant")
	if err != nil {
		return ""
	}
	out, err := exec.Command(antPath, "auth", "print-credentials", "--access-token").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func queryClaudeAPI(cfg providerConfig, prompt string) (string, error) {
	// Prepare request body
	reqBody := ClaudeRequest{
		Model:     cfg.model,
		MaxTokens: 1000,
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", claudeAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Set headers. OAuth tokens use Authorization: Bearer (plus a beta header
	// the API requires for OAuth on /v1/messages); API keys use x-api-key.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	switch cfg.claudeAuth {
	case authOAuth:
		req.Header.Set("Authorization", "Bearer "+cfg.apiKey)
		req.Header.Set("anthropic-beta", "oauth-2025-04-20")
	case authBearer:
		req.Header.Set("Authorization", "Bearer "+cfg.apiKey)
	default:
		req.Header.Set("x-api-key", cfg.apiKey)
	}

	// Make the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %v", err)
	}

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var claudeResp ClaudeResponse
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	// Check for API errors
	if claudeResp.Error != nil {
		return "", fmt.Errorf("API error: %s", claudeResp.Error.Message)
	}

	// Extract the text from response, skipping thinking blocks (which
	// appear first on models with adaptive thinking)
	var text strings.Builder
	for _, block := range claudeResp.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}

	command := strings.TrimSpace(text.String())
	if command == "" {
		return "", fmt.Errorf("no text in response")
	}

	return command, nil
}

func queryOpenAIAPI(apiKey, prompt string) (string, error) {
	return queryOpenAICompat(openaiAPIURL, apiKey, "gpt-4o-mini", prompt)
}

func queryLocalAPI(cfg providerConfig, prompt string) (string, error) {
	url := strings.TrimRight(cfg.baseURL, "/") + "/chat/completions"
	return queryOpenAICompat(url, cfg.apiKey, cfg.model, prompt)
}

func queryOpenAICompat(url, apiKey, model, prompt string) (string, error) {
	// Prepare request body
	reqBody := OpenAIRequest{
		Model:       model,
		MaxTokens:   1000,
		Temperature: 0.1,
		Messages: []OpenAIMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	// Make the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %v", err)
	}

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var openaiResp OpenAIResponse
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	// Check for API errors
	if openaiResp.Error != nil {
		return "", fmt.Errorf("API error: %s", openaiResp.Error.Message)
	}

	// Extract the command from response
	if len(openaiResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	command := strings.TrimSpace(openaiResp.Choices[0].Message.Content)
	if command == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return command, nil
}

func queryOllamaAPI(model, prompt string) (string, error) {
	// Prepare request body
	reqBody := OllamaRequest{
		Model:    model,
		Prompt:   prompt,
		Stream:   false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", ollamaAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")

	// Make the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %v", err)
	}

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var ollamaResp OllamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	// Check for API errors
	if ollamaResp.Error != nil {
		return "", fmt.Errorf("API error: %s", ollamaResp.Error.Message)
	}

	// Extract the command from response
	if ollamaResp.Response == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return strings.TrimSpace(ollamaResp.Response), nil

}

// ANSI escape codes for terminal formatting
const (
	Reset     = "\033[0m"
	Bold      = "\033[1m"
	Dim       = "\033[2m"
	Italic    = "\033[3m"
	Underline = "\033[4m"
	Red       = "\033[31m"
	Green     = "\033[32m"
	Yellow    = "\033[33m"
	Blue      = "\033[34m"
	Magenta   = "\033[35m"
	Cyan      = "\033[36m"
)

// RenderMarkdown converts basic markdown to terminal-formatted text
func RenderMarkdown(markdown string) string {
	lines := strings.Split(markdown, "\n")
	var result strings.Builder

	for _, line := range lines {
		rendered := renderLine(line)
		result.WriteString(rendered + "\n")
	}

	return strings.TrimSuffix(result.String(), "\n")
}

func renderLine(line string) string {
	// Handle headers
	if strings.HasPrefix(line, "### ") {
		return Yellow + Bold + strings.TrimPrefix(line, "### ") + Reset
	}
	if strings.HasPrefix(line, "## ") {
		return Blue + Bold + strings.TrimPrefix(line, "## ") + Reset
	}
	if strings.HasPrefix(line, "# ") {
		return Magenta + Bold + strings.TrimPrefix(line, "# ") + Reset
	}

	// Handle code blocks (simple single-line detection)
	if strings.HasPrefix(line, "```") {
		return Cyan + line + Reset
	}

	// Handle bullet points
	if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
		return Green + "• " + Reset + strings.TrimPrefix(strings.TrimPrefix(line, "- "), "* ")
	}

	// Handle numbered lists
	if matched, _ := regexp.MatchString(`^\d+\. `, line); matched {
		re := regexp.MustCompile(`^(\d+\. )(.*)`)
		matches := re.FindStringSubmatch(line)
		if len(matches) == 3 {
			return Yellow + matches[1] + Reset + matches[2]
		}
	}

	// Handle inline formatting
	line = renderInlineFormatting(line)

	return line
}

func renderInlineFormatting(text string) string {
	// Process bold first (**text** and __text__) to avoid conflicts with italic
	boldRe := regexp.MustCompile(`\*\*([^\*\n]*?)\*\*`)
	text = boldRe.ReplaceAllString(text, Bold+"$1"+Reset)

	boldRe2 := regexp.MustCompile(`__([^_\n]*?)__`)
	text = boldRe2.ReplaceAllString(text, Bold+"$1"+Reset)

	// Then process italic (*text* and _text_)
	// Use non-greedy matching and allow whitespace
	italicRe := regexp.MustCompile(`\*([^\*\n]*?)\*`)
	text = italicRe.ReplaceAllString(text, Italic+"$1"+Reset)

	italicRe2 := regexp.MustCompile(`_([^_\n]*?)_`)
	text = italicRe2.ReplaceAllString(text, Italic+"$1"+Reset)

	// Inline code (`code`) - preserve whitespace
	codeRe := regexp.MustCompile("`([^`\n]*?)`")
	text = codeRe.ReplaceAllString(text, Cyan+"$1"+Reset)

	// Links [text](url) - preserve whitespace
	linkRe := regexp.MustCompile(`\[([^\]\n]*?)\]\([^)\n]*?\)`)
	text = linkRe.ReplaceAllString(text, Blue+Underline+"$1"+Reset)

	return text
}
