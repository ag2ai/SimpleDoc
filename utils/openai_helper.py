from autogen import OpenAIWrapper, LLMConfig

def initialize_client(args, base_url=None, model_name=None):
    """Initialize the OpenAI client wrapper with the config list."""
    api_key = ""
    try:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    except Exception as e:
        print(f"Error reading API key file: {e}")
        return None
    
    config_list = [
        {
            "model": model_name if model_name else args.model,
            "base_url": base_url if base_url else args.base_url,
            "api_key": api_key,
            "api_type": "openai",
            "price": [0.08/1000, 0.24/1000]
        }
    ]
    
    return OpenAIWrapper(config_list=config_list, cache_seed=args.cache_seed)
