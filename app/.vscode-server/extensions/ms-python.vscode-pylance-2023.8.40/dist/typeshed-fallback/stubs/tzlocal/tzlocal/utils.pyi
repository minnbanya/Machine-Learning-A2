import datetime

def get_tz_offset(tz: datetime.tzinfo) -> int: ...
def assert_tz_offset(tz: datetime.tzinfo, error: bool = True) -> None: ...
