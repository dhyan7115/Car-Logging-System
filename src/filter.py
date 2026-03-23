import time
_last = {}
COOLDOWN = 8  # seconds

def is_allowed(plate):
    t = time.time()
    if plate not in _last or (t - _last[plate]) > COOLDOWN:
        _last[plate] = t
        return True
    return False