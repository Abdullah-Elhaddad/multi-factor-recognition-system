"""
create_user.py
- Ask for username and PIN
- Save bcrypt hash to: dataset/<username>/encrypted_pin.yaml
"""

import bcrypt, yaml
from pathlib import Path
from getpass import getpass
from datetime import datetime

DATASET_ROOT = Path("dataset")

def main():
    username = input("Enter username (folder-safe): ").strip()
    if not username:
        print("[ERR] Username is required."); return

    pin1 = getpass("Enter PIN (will be hashed): ").strip()
    pin2 = getpass("Re-enter PIN: ").strip()
    if not pin1 or pin1 != pin2:
        print("[ERR] Empty PIN or mismatch."); return

    user_dir = DATASET_ROOT / username
    user_dir.mkdir(parents=True, exist_ok=True)

    hashed = bcrypt.hashpw(pin1.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    out = {
        "username": username,
        "bcrypt_pin_hash": hashed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Verify later with bcrypt.checkpw(pin.encode(), stored_hash.encode())"
    }

    yaml_path = user_dir / "encrypted_pin.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"[OK] Saved -> {yaml_path}")

if __name__ == "__main__":
    main()
