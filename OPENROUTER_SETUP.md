# OpenRouter Setup Guide

## Issue: Free Models Require Privacy Configuration

OpenRouter's free models (those ending with `:free`) require you to configure data privacy settings before use.

### Error Message:
```
Error code: 404 - {'error': {'message': 'No endpoints found matching your data policy (Free model publication). Configure: https://openrouter.ai/settings/privacy', 'code': 404}}
```

## Solutions

### Option 1: Configure Privacy Settings (Recommended for Free Models)

1. Go to https://openrouter.ai/settings/privacy
2. Enable "Allow free model publication" or adjust your data policy
3. Save settings
4. You can now use free models like:
   - `openai/gpt-oss-120b:free`
   - `google/gemini-2.0-flash-exp:free`
   - `meta-llama/llama-3.2-3b-instruct:free`

### Option 2: Use Models Without Privacy Restrictions

Add to your `.env` file:

```env
# Models that work without privacy configuration
OPENROUTER_MODEL=qwen/qwen-2-7b-instruct:free
# or
OPENROUTER_MODEL=nousresearch/hermes-3-llama-3.1-405b:free
```

### Option 3: Use Paid Models (No Restrictions)

If you have credits in your OpenRouter account:

```env
OPENROUTER_MODEL=openai/gpt-3.5-turbo
# or
OPENROUTER_MODEL=anthropic/claude-3-haiku
```

## Using Reasoning Models

For models that support reasoning (like `openai/gpt-oss-120b:free`):

### Basic Usage (LangChain):

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free",
    temperature=0.1,
    model_kwargs={
        "extra_body": {
            "reasoning": {"enabled": True}
        }
    }
)

response = await llm.ainvoke("How many r's are in the word 'strawberry'?")
```

### Advanced Usage (Direct OpenAI Client):

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>",
)

# First API call with reasoning
response = client.chat.completions.create(
    model="openai/gpt-oss-120b:free",
    messages=[
        {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
        }
    ],
    extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response_msg = response.choices[0].message

# Continue conversation with reasoning context
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": response_msg.content,
        "reasoning_details": response_msg.reasoning_details  # Preserve reasoning
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning
response2 = client.chat.completions.create(
    model="openai/gpt-oss-120b:free",
    messages=messages,
    extra_body={"reasoning": {"enabled": True}}
)
```

## Recommended Free Models (After Privacy Setup)

### For General Use:
- `qwen/qwen-2-7b-instruct:free` - Good balance, works without privacy config
- `meta-llama/llama-3.2-3b-instruct:free` - Fast, lightweight
- `nousresearch/hermes-3-llama-3.1-405b:free` - High quality

### For Reasoning Tasks:
- `openai/gpt-oss-120b:free` - Supports reasoning API
- `google/gemini-2.0-flash-exp:free` - Fast and capable

### For Medical/Healthcare:
- `meta-llama/llama-3.1-70b-instruct:free` - Good for complex tasks
- `google/gemini-pro-1.5:free` - Multimodal support

## Environment Variables

Add these to your `.env` file:

```env
# Required
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx

# Optional (with defaults)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen-2-7b-instruct:free
OPENROUTER_EMBED_MODEL=text-embedding-3-small
```

## Testing Your Configuration

Run the test script:

```bash
python test_providers.py
```

This will verify:
- ✓ API key is valid
- ✓ Model is accessible
- ✓ LLM responses work correctly

## Current Fallback Chain

The application tries providers in this order:

1. **Gemini** (if `GEMINI_API_KEY` is set)
2. **OpenRouter** (if `OPENROUTER_API_KEY` is set)
3. **GitHub Models** (if `GITHUB_TOKEN` is set)

At least one provider must be working for the application to start.

## Troubleshooting

### Rate Limiting
```
Error code: 429 - Provider returned error
```
**Solution:** Wait a few minutes or switch to a different model

### Invalid API Key
```
Error code: 401 - User not found
```
**Solution:** Check your API key at https://openrouter.ai/keys

### Model Not Found
```
Error code: 404 - Model not found
```
**Solution:** Verify the model name at https://openrouter.ai/models

### Privacy Policy Error
```
Error code: 404 - No endpoints found matching your data policy
```
**Solution:** Configure privacy settings at https://openrouter.ai/settings/privacy
