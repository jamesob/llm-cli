package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
	"regexp"
)

const (
	claudeAPIURL = "https://api.anthropic.com/v1/messages"
	openaiAPIURL = "https://api.openai.com/v1/chat/completions"
	ollamaAPIURL = "http://localhost:11434/api/generate"
	version      = "1.0.0"
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
	Ollama
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	// Determine which API to use
	provider, apiKey, err := determineAPIProvider()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintf(os.Stderr, "Set one of the following environment variables:\n")
		fmt.Fprintf(os.Stderr, "  export ANTHROPIC_API_KEY=your_claude_api_key\n")
		fmt.Fprintf(os.Stderr, "  export OPENAI_API_KEY=your_openai_api_key\n")
		os.Exit(1)
	}

	// Define flags
	var codeMode bool
	var explainMode bool
	
	// Custom flag set to handle both short and long flags
	flagSet := flag.NewFlagSet("llm", flag.ExitOnError)
	flagSet.BoolVar(&codeMode, "code", false, "Code generation mode")
	flagSet.BoolVar(&codeMode, "c", false, "Code generation mode (short)")
	flagSet.BoolVar(&explainMode, "explain", false, "Explanation mode")
	flagSet.BoolVar(&explainMode, "x", false, "Explanation mode (short)")
	
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
	err = flagSet.Parse(os.Args[1:])
	if err != nil {
		os.Exit(1)
	}
	
	query := strings.Join(flagSet.Args(), " ")

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
	switch provider {
	case Claude:
		response, err = queryClaudeAPI(apiKey, prompt)
	case OpenAI:
		response, err = queryOpenAIAPI(apiKey, prompt)
	case Ollama:
		response, err = queryOllamaAPI(apiKey, prompt)
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
    Set one of the following environment variables:
    export ANTHROPIC_API_KEY=your_claude_api_key
    export OPENAI_API_KEY=your_openai_api_key
    export OLLAMA_MODEL=your_ollama_model_name

    The script will automatically detect which API key or Ollama model is available and use the corresponding service.
    Priority order: Claude > OpenAI > Ollama

OPTIONS:
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

func determineAPIProvider() (APIProvider, string, error) {
	// Check for Claude API key first
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		return Claude, apiKey, nil
	}

	// Check for OpenAI API key
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		return OpenAI, apiKey, nil
	}

	// Check for Ollama model
	if model := os.Getenv("OLLAMA_MODEL"); model != "" {
		return Ollama, model, nil
	}

	return Claude, "", fmt.Errorf("no API key or Ollama model found")
}

func queryClaudeAPI(apiKey, prompt string) (string, error) {
	// Prepare request body
	reqBody := ClaudeRequest{
		Model:     "claude-sonnet-4-20250514",
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

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

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

	// Extract the command from response
	if len(claudeResp.Content) == 0 {
		return "", fmt.Errorf("no content in response")
	}

	command := strings.TrimSpace(claudeResp.Content[0].Text)
	if command == "" {
		return "", fmt.Errorf("empty response from API")
	}

	return command, nil
}

func queryOpenAIAPI(apiKey, prompt string) (string, error) {
	// Prepare request body
	reqBody := OpenAIRequest{
		Model:       "gpt-4o-mini",
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
	req, err := http.NewRequest("POST", openaiAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

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
