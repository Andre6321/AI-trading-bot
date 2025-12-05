def calculate_percentage_change(old_value, new_value):
    if old_value == 0:
        raise ValueError("Old value cannot be zero for percentage change calculation.")
    return ((new_value - old_value) / old_value) * 100

def format_currency(value, currency_symbol="$"):
    return f"{currency_symbol}{value:,.2f}"

def validate_api_key(api_key):
    if not isinstance(api_key, str) or len(api_key) == 0:
        raise ValueError("Invalid API key. It must be a non-empty string.")

def log_message(message, level="INFO"):
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level not in levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {levels}.")
    print(f"{level}: {message}")