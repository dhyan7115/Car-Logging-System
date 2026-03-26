import pandas as pd
from datetime import datetime, timedelta
import os
import hashlib

FILE = "/home/dhyan/Desktop/DL/log.xlsx"

# ✅ FIX Bug 4: Debounce — ignore same plate within this many seconds
DEBOUNCE_SECONDS = 10


class VehicleLogger:
    def __init__(self, file_path=FILE):
        self.file_path = file_path
        self._last_seen = {}  # plate -> last log timestamp (for debounce)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create Excel file if it doesn't exist"""
        # ✅ FIX Bug 1: Only call makedirs if there's actually a directory component
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if not os.path.exists(self.file_path):
            print(f"📁 Creating new log file: {self.file_path}")
            df = pd.DataFrame(columns=[
                "Plate", "Entry_Time", "Exit_Time",
                "Duration", "Status", "Session_ID"
            ])
            df.to_excel(self.file_path, index=False, engine='openpyxl')

    def _read_excel(self):
        """Read Excel file with explicit engine."""
        try:
            return pd.read_excel(self.file_path, engine='openpyxl', dtype=str)
        except Exception as e:
            print(f"⚠️ Error reading Excel: {e}")
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            self._ensure_file_exists()
            return pd.read_excel(self.file_path, engine='openpyxl', dtype=str)

    def _write_excel(self, df):
        """Write Excel file with explicit engine"""
        df.to_excel(self.file_path, index=False, engine='openpyxl')

    def _get_active_session(self, plate):
        """Get active session for a plate (entry without exit)"""
        df = self._read_excel()
        active = df[(df["Plate"] == plate) & (df["Exit_Time"].isna())]
        if not active.empty:
            return active.iloc[-1]
        return None

    def _is_debounced(self, plate):
        """Return True if this plate was logged too recently"""
        if plate in self._last_seen:
            elapsed = (datetime.now() - self._last_seen[plate]).total_seconds()
            if elapsed < DEBOUNCE_SECONDS:
                print(f"⏱️ Debounced: {plate} was logged {elapsed:.1f}s ago, skipping")
                return True
        return False

    def log_entry(self, plate):
        """Log vehicle entry"""
        if self._is_debounced(plate):
            return False

        print(f"📝 [ENTRY] Processing entry for: {plate}")
        df = self._read_excel()
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        session_id = hashlib.md5(f"{plate}_{now.timestamp()}".encode()).hexdigest()[:8]

        active = self._get_active_session(plate)
        if active is not None:
            print(f"⚠️ {plate} already has an open session since {active['Entry_Time']}")
            print("   Auto-closing previous session before logging new entry")
            idx = active.name
            entry_time = pd.to_datetime(active["Entry_Time"])
            duration = now - entry_time.to_pydatetime()
            # Store as strings — avoids all dtype/precision issues
            df.loc[idx, "Exit_Time"] = now_str
            df.loc[idx, "Duration"] = str(duration).split('.')[0]
            df.loc[idx, "Status"] = "Exited"
            self._write_excel(df)
            df = self._read_excel()

        new_row = {
            "Plate": plate,
            "Entry_Time": now_str,
            "Exit_Time": None,
            "Duration": None,
            "Status": "Inside",
            "Session_ID": session_id
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self._write_excel(df)

        self._last_seen[plate] = now
        print(f"✅ ENTRY LOGGED: {plate} at {now_str}")
        return True

    def log_exit(self, plate):
        """Log vehicle exit"""
        if self._is_debounced(plate):
            return False

        print(f"📝 [EXIT] Processing exit for: {plate}")
        df = self._read_excel()
        active = self._get_active_session(plate)

        if active is None:
            print(f"❌ No active session found for {plate}. Was entry missed?")
            return False

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        entry_time = pd.to_datetime(active["Entry_Time"]).to_pydatetime()
        duration = now - entry_time

        idx = active.name
        df.loc[idx, "Exit_Time"] = now_str
        df.loc[idx, "Duration"] = str(duration).split('.')[0]
        df.loc[idx, "Status"] = "Exited"
        self._write_excel(df)

        self._last_seen[plate] = now
        print(f"✅ EXIT LOGGED: {plate}")
        print(f"   Entry:    {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Exit:     {now_str}")
        print(f"   Duration: {str(duration).split('.')[0]}")
        return True

    def get_status(self, plate=None):
        """Get current status of vehicle(s)"""
        try:
            if not os.path.exists(self.file_path):
                return [] if plate is None else "Outside"

            df = self._read_excel()

            if plate:
                # ✅ FIX Bug 3: Single source of truth — Exit_Time only
                active = self._get_active_session(plate)
                return "Inside" if active is not None else "Outside"
            else:
                active_vehicles = df[df["Exit_Time"].isna()]
                return active_vehicles["Plate"].tolist()

        except Exception as e:
            print(f"⚠️ Error getting status: {e}")
            return [] if plate is None else "Unknown"


# Global instance
_logger = VehicleLogger()


def log_plate(plate, direction="entry"):
    """
    direction: "entry" or "exit"
    """
    if direction == "entry":
        return _logger.log_entry(plate)
    elif direction == "exit":
        return _logger.log_exit(plate)
    else:
        print(f"❌ Invalid direction: {direction}")
        return False