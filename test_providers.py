"""
Test script to verify provider configurations and API tokens.
Run this to check if your .env file has correct tokens and model configurations.
"""

import os
import asyncio
from dotenv import load_dotenv

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ {text}{RESET}")

async def test_gemini_provider():
    """Test Gemini provider configuration."""
    print_header("Testing Gemini Provider")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print_error("No Gemini API key found")
        print_info("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env file")
        return False
    
    print_success(f"API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Test LLM
        print_info("Testing Gemini LLM...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        response = await llm.ainvoke("Say 'Hello' in one word")
        print_success(f"LLM Response: {response.content}")
        
        return True
        
    except Exception as e:
        print_error(f"Gemini test failed: {str(e)[:100]}")
        return False

async def test_openrouter_provider():
    """Test OpenRouter provider configuration."""
    print_header("Testing OpenRouter Provider")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print_error("No OpenRouter API key found")
        print_info("Set OPENROUTER_API_KEY in .env file")
        return False
    
    print_success(f"API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2-7b-instruct:free")
    embed_model = os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small")
    
    print_info(f"Base URL: {base_url}")
    print_info(f"LLM Model: {model}")
    print_info(f"Embedding Model: {embed_model}")
    print_info("Note: Free models may require privacy settings at https://openrouter.ai/settings/privacy")
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # Test LLM
        print_info("Testing OpenRouter LLM...")
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.1,
            model_kwargs={"extra_body": {"reasoning": {"enabled": False}}}  # Disable reasoning for simple test
        )
        response = await llm.ainvoke("Say 'Hello' in one word")
        print_success(f"LLM Response: {response.content}")
        
        return True
        
    except Exception as e:
        print_error(f"OpenRouter test failed: {str(e)[:200]}")
        return False

async def test_github_provider():
    """Test GitHub Models provider configuration."""
    print_header("Testing GitHub Models Provider")
    
    token = os.getenv("GITHUB_TOKEN")
    
    if not token:
        print_error("No GitHub token found")
        print_info("Set GITHUB_TOKEN in .env file")
        return False
    
    print_success(f"Token found: {token[:10]}...{token[-4:]}")
    
    base_url = os.getenv("GITHUB_OPENAI_BASE_URL", "https://models.inference.ai.azure.com")
    model = os.getenv("GITHUB_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("GITHUB_EMBED_MODEL", "text-embedding-3-small")
    
    print_info(f"Base URL: {base_url}")
    print_info(f"LLM Model: {model}")
    print_info(f"Embedding Model: {embed_model}")
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # Test LLM
        print_info("Testing GitHub LLM...")
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=token,
            model=model,
            temperature=0.1
        )
        response = await llm.ainvoke("Say 'Hello' in one word")
        print_success(f"LLM Response: {response.content}")
        
        return True
        
    except Exception as e:
        print_error(f"GitHub test failed: {str(e)[:200]}")
        return False

async def main():
    """Run all provider tests."""
    load_dotenv()
    
    print_header("Provider Configuration Test")
    print_info("This script tests your .env configuration for all providers\n")
    
    results = {
        "Gemini": await test_gemini_provider(),
        "OpenRouter": await test_openrouter_provider(),
        "GitHub": await test_github_provider()
    }
    
    # Summary
    print_header("Test Summary")
    
    working_providers = [name for name, status in results.items() if status]
    failed_providers = [name for name, status in results.items() if not status]
    
    if working_providers:
        print_success(f"Working providers: {', '.join(working_providers)}")
    
    if failed_providers:
        print_error(f"Failed providers: {', '.join(failed_providers)}")
    
    print()
    if len(working_providers) == 0:
        print_error("❌ No providers are working! Please check your .env file.")
        print_info("\nRequired environment variables:")
        print_info("  Gemini: GEMINI_API_KEY or GOOGLE_API_KEY")
        print_info("  OpenRouter: OPENROUTER_API_KEY")
        print_info("  GitHub: GITHUB_TOKEN")
    elif len(working_providers) < 3:
        print_warning(f"⚠ Only {len(working_providers)}/3 providers working. Fallback is available but limited.")
    else:
        print_success("✅ All providers working! Full fallback chain available.")
    
    print()

if __name__ == "__main__":
    asyncio.run(main())
