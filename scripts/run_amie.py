"""
Phase 2 of AMIE3 Horn Clause Mining Pipeline
=============================================
Runs the AMIE3 JAR against the exported TSV and captures the raw output.
Then immediately pipes it into Phase 3 parsing via parse_amie_output.py.

Configuration flags explained:
  -mins  Minimum support threshold (how many graph triples must back the rule)
  -minc  Minimum confidence threshold (0.0 - 1.0)
  -maxad Maximum number of atoms in the rule body (rule complexity)
  -hc    Minimum head coverage
  -Xmx   Java heap size (adjust based on your system RAM)

Usage:
    uv run python scripts/run_amie.py
"""

import subprocess
import sys
import os
from pathlib import Path

AMIE_JAR      = "amie/amie3.jar"
INPUT_TSV     = "data/amie_input.tsv"
RAW_OUTPUT    = "outputs/amie_raw_output.txt"

# Use the hardcoded path on Windows, or default to 'java' on the Linux cluster
JAVA_EXECUTABLE = r"C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot\bin\java.exe" if os.name == 'nt' else "java"

# ── Tuning knobs ────────────────────────────────────────────────────────────
# With only 14K facts, support=100 mines nothing. Use low thresholds.
# 14,285 facts means support=3 is ~0.02% — that's already very selective.
MIN_SUPPORT        = 2      # must appear at least 2 times
MIN_CONFIDENCE     = 0.1    # at least 10% confidence
MIN_HEAD_COVERAGE  = 0.01   # rule must explain at least 1% of head relation
MAX_RULE_LENGTH    = 3      # 3 atoms (2-hop body + 1-hop head)
JAVA_HEAP_GB       = 64     # 64GB for cluster execution
# ────────────────────────────────────────────────────────────────────────────

def run_amie():
    # Sanity checks
    if not Path(AMIE_JAR).exists():
        print(f"[ERROR] AMIE3 JAR not found at {AMIE_JAR}")
        print("  Run: New-Item -ItemType Directory -Force amie")
        print("  Then download: https://github.com/dig-team/AMIE/releases")
        sys.exit(1)

    if not Path(INPUT_TSV).exists():
        print(f"[ERROR] Input TSV not found at {INPUT_TSV}")
        print("  Run: uv run python scripts/export_for_amie.py")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)

    cmd = [
        JAVA_EXECUTABLE,
        f"-Xmx{JAVA_HEAP_GB}G",
        "-jar", AMIE_JAR,
        "-mins",  str(MIN_SUPPORT),
        "-minc",  str(MIN_CONFIDENCE),
        "-minhc", str(MIN_HEAD_COVERAGE),
        "-maxad", str(MAX_RULE_LENGTH),
        INPUT_TSV,
    ]

    print("\n[AMIE3] Starting rule mining...")
    print(f"  Input:      {INPUT_TSV}")
    print(f"  Min support:     {MIN_SUPPORT}")
    print(f"  Min confidence:  {MIN_CONFIDENCE}")
    print(f"  Max rule length: {MAX_RULE_LENGTH}")
    print(f"  Java heap:       {JAVA_HEAP_GB}GB")
    print(f"\n  Command: {' '.join(cmd)}\n")
    print("  This may take several minutes for large graphs...\n")

    with open(RAW_OUTPUT, "w", encoding="utf-8") as out_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            print(line, end="")   # live stream to terminal
            out_f.write(line)     # also save to file

        process.wait()

    if process.returncode != 0:
        print(f"\n[ERROR] AMIE3 exited with code {process.returncode}")
        sys.exit(process.returncode)

    print(f"\n[AMIE3] Mining complete. Raw output saved to: {RAW_OUTPUT}")
    print("\nNext step -> parse the output:")
    print("  uv run python scripts/parse_amie_output.py")


if __name__ == "__main__":
    run_amie()
