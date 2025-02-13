from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))
