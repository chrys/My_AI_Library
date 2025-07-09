import configparser
import os

def read_config(section: str, key: str) -> str:
    # For production, read from environment variables
    value = os.getenv(key)
    if value is None:
        raise KeyError(f"Environment variable '{key}' not found")
    return value
     

if __name__ == "__main__":
