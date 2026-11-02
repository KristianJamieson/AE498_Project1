from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def inspect_directory(directory, max_files=12):
    path = PROJECT_ROOT / directory

    print(f"\n{'=' * 65}")
    print(f"Directory: {path}")
    print("=" * 65)

    if not path.exists():
        print("NOT FOUND")
        return

    files = sorted(p for p in path.rglob("*") if p.is_file())

    print(f"Total files: {len(files)}")

    for file in files[:max_files]:
        relative_path = file.relative_to(PROJECT_ROOT)
        size_mb = file.stat().st_size / (1024 ** 2)
        print(f"{relative_path}  ({size_mb:.3f} MB)")

    if len(files) > max_files:
        print(f"... and {len(files) - max_files} more files")

for directory in ["initial_states", "omni2", "sat_density"]:
    inspect_directory(directory)
