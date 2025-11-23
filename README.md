# Overspecifier

A prompt modifier and attribute verbosizer that overspecifies and enhances the specificity of your image generation prompts—unlocking fine-grained control and a richer user experience.

## Quickstart

1. Install `uv` and then:
    ```bash
    uv init --bare --python 3.12
    uv sync --python 3.12
    source .venv/bin/activate
    ```
2. Run `main.py` after specifying the model and attribute expansion arguments.

## File Descriptions

### Core Files
- **`overspecifier.py`** - Main Overspecifier class for prompt tokenization and attribute expansion
- **`prompts.py`** - Prompt templates and system messages for LLM calls
- **`main.py`** - CLI script to process CSV files and generate JSONL outputs

### Input Format (`csv/*.csv`)
CSV with columns:
- `Example` - The original prompt text

### Output Format (`results/*.jsonl`)
JSONL where each line contains:
```json
{
  "prompt": "original prompt text",
  "attr_per_token": [
    {
      "text": "token_word",
      "is_modifiable": true/false,
      "attributes": [
        {
          "name": "attribute_name",
          "values": ["value1", "value2", ...],
          "is_contrasting": true/false
        }
      ]
    }
  ]
}
```


