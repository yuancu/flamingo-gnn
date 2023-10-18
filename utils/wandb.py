import os

# Suggest a better name for this function

def set_wandb_api_key_from_file(token_path='.wandbtoken'):
    """Set WANDB_API_KEY environment variable to the value in the token file.
    """
    # if '.wandbtoken' file exists, read it and set WANDB_API_KEY to it
    if os.path.exists(token_path):
        with open(token_path, encoding='utf-8') as f:
            wandb_api_key = f.read().strip()
            print(f"Setting WANDB_API_KEY to {wandb_api_key}")
            os.environ['WANDB_API_KEY'] = wandb_api_key
    else:
        print("Using default wandb setting.")
