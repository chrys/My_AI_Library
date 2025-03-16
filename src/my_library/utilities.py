import configparser
import os

def read_config(section: str, key: str) -> str:
    """
    Read sensitive data from the configuration file ~/.config.ini.

    Business Requirement:
    - Read sensitive data securely.

    Technical Requirements:
    - Read sensitive data from file ~/.config.ini in the form of:
      [Section]
      key = value

    Args:
        section (str): The section in the config file.
        key (str): The key within the section to retrieve the value for.

    Returns:
        str: The value associated with the given section and key.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the section or key is not found in the config file.
    """
    config_path = os.path.expanduser('~/.config.ini')
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Initialize the config parser
    config = configparser.ConfigParser()
    
    # Read the config file
    config.read(config_path)
    
    # Retrieve the value from the specified section and key
    try:
        value = config[section][key]
    except KeyError as e:
        raise KeyError(f"Section or key not found in config file: {e}")
    
    return value

if __name__ == "__main__":
    read_config("AI KEYS", "openai")