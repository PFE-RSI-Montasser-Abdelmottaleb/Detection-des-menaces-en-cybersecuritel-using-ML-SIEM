import json
import random
import time
import os
from datetime import datetime, timezone


LOG_FILE = "logs/testSimulate.json"
# Valeurs catégoriques alignées avec app.py
protocols = ["tcp", "udp", "icmp"]  # Correspond à protocol_type_mapping
services = ["http", "ftp", "smtp"]    # Correspond à service_mapping
flags = ["SF", "REJ", "RSTO"]        # Correspond à flag_mapping
def generate_log_entry():
    entry = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "duration": random.randint(0, 100),
        "protocol_type": random.choice(protocols),
        "service": random.choice(services),
        "flag": random.choice(flags),
        "src_bytes": random.randint(0, 1500),
        "dst_bytes": random.randint(0, 5000),
        "wrong_fragment": 0,
        "hot": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "count": random.randint(1, 100),
        "srv_count": random.randint(1, 100),
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0
    }
    return entry

def generate_attack_log():
    return {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "duration": random.randint(0, 100),
        "protocol_type": "tcp",
        "service": "http",
        "flag": "REJ",
        "src_bytes": random.randint(0, 100),
        "dst_bytes": random.randint(0, 100),
        "wrong_fragment": 0,
        "hot": 0,
        "logged_in": 0,
        "num_compromised": random.randint(0, 5),
        "count": random.randint(80, 100),
        "srv_count": random.randint(80, 100),
        "serror_rate": 1.0,
        "srv_serror_rate": 1.0,
        "rerror_rate": 0.0
    }


def main():
    print(f"[SIMULATEUR] Écriture de logs compatibles avec l'API ML dans {LOG_FILE}...")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    while True:
        entry = generate_attack_log() if random.random() < 0.3 else generate_log_entry()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print("[+] Log généré:", entry)
        time.sleep(1)


if __name__ == "__main__":
    main()