import pandas as pd
from datetime import datetime
import os

# Absolute path (IMPORTANT)
FILE = "/home/dhyan/Desktop/DL/log.xlsx"

def log_plate(plate):
    print("📝 Logging plate:", plate)

    # Create file if not exists
    if not os.path.exists(FILE):
        print("📄 Creating new Excel file...")
        df = pd.DataFrame(columns=["Plate", "Entry", "Exit"])
        df.to_excel(FILE, index=False)

    # Load file
    df = pd.read_excel(FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Entry / Exit logic
    if plate not in df["Plate"].values:
        df.loc[len(df)] = [plate, now, ""]
        print(f"[ENTRY] {plate}")
    else:
        idx = df[(df["Plate"] == plate) & (df["Exit"] == "")].index
        if len(idx) > 0:
            df.loc[idx[0], "Exit"] = now
            print(f"[EXIT] {plate}")
        else:
            df.loc[len(df)] = [plate, now, ""]
            print(f"[RE-ENTRY] {plate}")

    df.to_excel(FILE, index=False)
    print("✅ Excel updated at:", FILE)