# Jailbreak the Vault — make.fun Competition Entry

A Python agent that social-engineers a guarded AI character into revealing a secret code, across three levels of increasing filter hardness.

## Strategy

- **Level 1** (no filter): Direct asks and authority persona messages
- **Level 2** (regex filter): Indirect elicitation — alphabet positions, NATO phonetic, acrostics
- **Level 3** (strict filter): Creative structural encodings — acrostic stories, shopping lists, riddles

After every reply, the agent scans for 6-character codes using regex and Claude-assisted extraction, then guesses immediately.

## Requirements

- Python 3.13
- Standard library only (no pip installs)
- Anthropic API key (set as `ANTHROPIC_API_KEY`)

## Environment Variables

| Variable | Description |
|---|---|
| `MAKEFUN_ENV_URL` | Vault chat endpoint |
| `MAKEFUN_PROMPT_FILE` | Path to character prompt |
| `MAKEFUN_TRIAL_SEED` | Seed for determinism |
| `ANTHROPIC_API_KEY` | Your Anthropic key |

## Usage

```bash
python main.py
```

## License

MIT
