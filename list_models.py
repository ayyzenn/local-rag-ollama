"""Utility script to list available Gemini models"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=api_key)

print("🔍 Listing available Gemini models...\n")

try:
    models = genai.list_models()
    
    print("Available models that support generateContent:")
    print("=" * 60)
    
    supported_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            supported_models.append(model.name)
            # Extract just the model identifier (last part after /)
            model_id = model.name.split('/')[-1] if '/' in model.name else model.name
            print(f"  ✓ {model_id}")
            if hasattr(model, 'display_name'):
                print(f"    Display Name: {model.display_name}")
            print(f"    Full Name: {model.name}")
            print()
    
    if supported_models:
        print("\n💡 Recommended model names for config.py:")
        print(f"  GEMINI_MODEL = \"{supported_models[0].split('/')[-1]}\"")
        if len(supported_models) > 1:
            print(f"  # Or alternatively:")
            for model_name in supported_models[1:3]:  # Show first 3
                print(f"  # GEMINI_MODEL = \"{model_name.split('/')[-1]}\"")
    else:
        print("❌ No models found that support generateContent")
        print("   Check your API key and API access permissions")

except Exception as e:
    print(f"❌ Error listing models: {str(e)}")
    print("   Make sure your API key is valid and has proper permissions")

