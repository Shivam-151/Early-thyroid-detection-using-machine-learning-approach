from pathlib import Path

file_path = Path("data/hypothyroid.data")

with file_path.open("r", encoding="utf-8", errors="ignore") as f:
    for i in range(10):
        line = f.readline().rstrip("\n")
        print(f"\nLINE {i+1}:")
        print(line)
        print("Comma count:", line.count(","))