import time


class PlateFilter:
    """
    Prevents duplicate detections within cooldown period.
    Also tracks direction for entry/exit logic.
    """
    def __init__(self, cooldown_seconds=5):
        self.cooldown = cooldown_seconds
        self.last_detection = {}   # plate -> timestamp
        self.last_direction = {}   # plate -> direction

    def is_allowed(self, plate, current_direction=None):
        """
        Check if plate can be processed based on cooldown and direction.
        Returns: (allowed: bool, direction_to_use: str | None)
        """
        current_time = time.time()

        if plate in self.last_detection:
            time_diff = current_time - self.last_detection[plate]

            if time_diff < self.cooldown:
                prev_direction = self.last_direction.get(plate)

                # Same direction within cooldown → ignore
                if prev_direction == current_direction:
                    print(f"⏱️ [{plate}] Ignored — same direction '{current_direction}' within cooldown ({time_diff:.1f}s)")
                    return False, None

                # ✅ FIX Bug 1: Different direction within cooldown → allow but
                # only update direction, NOT the timestamp, so the cooldown
                # clock keeps running from the original detection
                print(f"🔄 [{plate}] Direction changed: '{prev_direction}' → '{current_direction}' (within cooldown)")
                self.last_direction[plate] = current_direction
                return True, current_direction

        # Outside cooldown or first detection → allow and reset timer
        self.last_detection[plate] = current_time
        self.last_direction[plate] = current_direction
        return True, current_direction

    def clear(self, plate=None):
        """Clear filter for a specific plate or all plates"""
        if plate:
            self.last_detection.pop(plate, None)
            self.last_direction.pop(plate, None)
        else:
            self.last_detection.clear()
            self.last_direction.clear()

    # ✅ FIX Bug 4: Debug helper to see current filter state
    def status(self):
        """Print current tracked plates and their cooldown remaining"""
        now = time.time()
        if not self.last_detection:
            print("📋 PlateFilter: No plates currently tracked")
            return

        print("📋 PlateFilter Status:")
        for plate, ts in self.last_detection.items():
            remaining = self.cooldown - (now - ts)
            direction = self.last_direction.get(plate, "unknown")
            if remaining > 0:
                print(f"   {plate} | direction={direction} | cooldown expires in {remaining:.1f}s")
            else:
                print(f"   {plate} | direction={direction} | cooldown expired")


# Global instance
# ✅ FIX Bug 2: Accept direction in the wrapper so direction logic is actually used
_filter = PlateFilter(cooldown_seconds=5)


def is_allowed(plate, direction=None):
    """
    Simple wrapper around PlateFilter.is_allowed().
    Pass direction="entry" or direction="exit" for correct filtering.
    Returns: (allowed: bool, direction: str | None)
    """
    # ✅ FIX Bug 2: direction is now passed through instead of hardcoded None
    return _filter.is_allowed(plate, current_direction=direction)