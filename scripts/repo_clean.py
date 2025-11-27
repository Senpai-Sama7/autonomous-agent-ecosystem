#!/usr/bin/env python3
"""
Repo Janitor: Safe, reversible cleanup with manifest and restore script.
Moves files to .trash/<timestamp>/ preserving relative structure.
"""
import os
import sys
import json
import hashlib
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

TRASH_BASE = ".trash"

def sha256_file(path: str) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def get_commit_date(path: str) -> str:
    """Get last commit date for a file (ISO format)."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", path],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def count_references(path: str, search_dir: str = ".") -> int:
    """Count inbound references to a file path."""
    try:
        result = subprocess.run(
            ["grep", "-r", "--include=*.py", "--include=*.md", 
             "--include=*.json", "--include=*.yaml", "-l", 
             os.path.basename(path), search_dir],
            capture_output=True, text=True
        )
        # Exclude self-references and this script
        refs = [r for r in result.stdout.strip().split("\n") 
                if r and r != path and ".trash" not in r 
                and "repo_clean" not in r and "scripts/" not in r]
        return len(refs)
    except Exception:
        return -1

def get_candidates() -> List[Dict[str, Any]]:
    """Identify files/dirs to move to trash."""
    candidates = []
    
    # 1. REALITY_CHECKLIST.txt - superseded
    if os.path.exists("REALITY_CHECKLIST.txt"):
        refs = count_references("REALITY_CHECKLIST.txt")
        if refs == 0:
            candidates.append({
                "original_path": "REALITY_CHECKLIST.txt",
                "size_bytes": os.path.getsize("REALITY_CHECKLIST.txt"),
                "sha256": sha256_file("REALITY_CHECKLIST.txt"),
                "reason": "superseded",
                "detected_by": ["no_backlinks", "older_commit_date"],
                "commit_date_iso": get_commit_date("REALITY_CHECKLIST.txt"),
                "notes": "superseded by FEATURES_CHECKLIST.txt (more complete)"
            })
    
    # 2. evidence/comprehensive_health_check.txt - superseded by v2
    old_hc = "evidence/comprehensive_health_check.txt"
    if os.path.exists(old_hc) and os.path.exists(old_hc.replace(".txt", "_v2.txt")):
        candidates.append({
            "original_path": old_hc,
            "size_bytes": os.path.getsize(old_hc),
            "sha256": sha256_file(old_hc),
            "reason": "superseded",
            "detected_by": ["v2_exists", "older_timestamp"],
            "commit_date_iso": get_commit_date(old_hc),
            "notes": "superseded by comprehensive_health_check_v2.txt"
        })
    
    # 3. Empty directories
    for empty_dir in [".qodo", "fixes"]:
        if os.path.isdir(empty_dir):
            # Check if truly empty (no files, only empty subdirs)
            has_files = any(
                os.path.isfile(os.path.join(root, f))
                for root, _, files in os.walk(empty_dir) for f in files
            )
            if not has_files:
                candidates.append({
                    "original_path": empty_dir,
                    "size_bytes": 0,
                    "sha256": "N/A",
                    "reason": "empty_dir",
                    "detected_by": ["no_files"],
                    "commit_date_iso": "N/A",
                    "notes": f"empty directory tree"
                })
    
    return candidates

def move_to_trash(candidates: List[Dict], trash_dir: str, dry_run: bool = True) -> Dict:
    """Move candidates to trash directory."""
    manifest = {
        "version": 1,
        "timestamp": datetime.now().isoformat(),
        "trash_dir": trash_dir,
        "moved": []
    }
    
    for item in candidates:
        orig = item["original_path"]
        new_path = os.path.join(trash_dir, orig)
        item["new_path"] = new_path
        
        if dry_run:
            print(f"[DRY-RUN] Would move: {orig} → {new_path}")
        else:
            os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
            if os.path.isdir(orig):
                shutil.move(orig, new_path)
            else:
                shutil.move(orig, new_path)
            print(f"[MOVED] {orig} → {new_path}")
        
        manifest["moved"].append(item)
    
    return manifest

def write_manifest(manifest: Dict, trash_dir: str):
    """Write manifest.json to trash directory."""
    manifest_path = os.path.join(trash_dir, "manifest.json")
    os.makedirs(trash_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[WROTE] {manifest_path}")

def write_restore_script(manifest: Dict, trash_dir: str):
    """Generate restore.sh script."""
    restore_path = os.path.join(trash_dir, "restore.sh")
    
    lines = [
        "#!/bin/bash",
        "# Restore script - reverses all moves from this cleanup",
        f"# Generated: {manifest['timestamp']}",
        "# Usage: bash restore.sh",
        "",
        "set -e",
        ""
    ]
    
    for item in manifest["moved"]:
        orig = item["original_path"]
        new = item["new_path"]
        if item["reason"] == "empty_dir":
            lines.append(f'mkdir -p "{orig}"')
            lines.append(f'rm -rf "{new}"')
        else:
            lines.append(f'mkdir -p "$(dirname "{orig}")"')
            lines.append(f'mv "{new}" "{orig}"')
    
    lines.append("")
    lines.append('echo "Restore complete. You may delete this .trash directory."')
    
    with open(restore_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(restore_path, 0o755)
    print(f"[WROTE] {restore_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Repo Janitor - safe cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved")
    parser.add_argument("--apply", action="store_true", help="Actually move files")
    parser.add_argument("--trash-dir", default=None, help="Custom trash directory")
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        sys.exit(1)
    
    trash_dir = args.trash_dir or os.path.join(
        TRASH_BASE, datetime.now().strftime("%Y%m%d-%H%M")
    )
    
    print(f"Repo Janitor - {'DRY RUN' if args.dry_run else 'APPLYING'}")
    print(f"Trash directory: {trash_dir}\n")
    
    candidates = get_candidates()
    
    if not candidates:
        print("No candidates found for cleanup.")
        return
    
    print(f"Found {len(candidates)} candidate(s):\n")
    for c in candidates:
        print(f"  - {c['original_path']} [{c['reason']}]")
    print()
    
    manifest = move_to_trash(candidates, trash_dir, dry_run=args.dry_run)
    
    if args.apply:
        write_manifest(manifest, trash_dir)
        write_restore_script(manifest, trash_dir)
        print("\n✅ Cleanup complete. Run `bash {}/restore.sh` to revert.".format(trash_dir))
    else:
        print("\n[DRY-RUN] No changes made. Use --apply to execute.")

if __name__ == "__main__":
    main()
