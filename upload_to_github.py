"""
Automated helper to initialize Git, configure remotes, and push this project to GitHub.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], check: bool = True) -> tuple[int, str]:
    """Runs a shell command and returns (returncode, stdout + stderr)."""
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=check,
            cwd=Path(__file__).resolve().parent
        )
        return res.returncode, res.stdout.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout.strip() if e.stdout else str(e)


def main():
    parser = argparse.ArgumentParser(description="Upload EchoPulse project to GitHub")
    parser.add_argument("--repo", "-r", type=str, default=None, help="GitHub repository URL (e.g. https://github.com/user/repo.git)")
    parser.add_argument("--branch", "-b", type=str, default="main", help="Target branch (default: main)")
    parser.add_argument("--user", type=str, default=None, help="Git user name")
    parser.add_argument("--email", type=str, default=None, help="Git user email")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    print(f"============================================================")
    print(f" EchoPulse AI — GitHub Upload & Synchronization Helper")
    print(f" Working Directory: {project_dir}")
    print(f"============================================================\n")

    # 1. Check Git
    code, out = run_cmd(["git", "--version"], check=False)
    if code != 0:
        print("[ERROR] Git is not installed or not in your PATH.")
        sys.exit(1)
    print(f"✓ Found: {out}")

    # 2. Check Git User / Email
    code_name, current_name = run_cmd(["git", "config", "user.name"], check=False)
    code_email, current_email = run_cmd(["git", "config", "user.email"], check=False)

    if code_name != 0 or not current_name:
        user_name = args.user or input("Enter your Git/GitHub Name: ").strip() or "AI Audio Developer"
        run_cmd(["git", "config", "user.name", user_name])
        print(f"✓ Configured git user.name: {user_name}")
    else:
        print(f"✓ Git user.name: {current_name}")

    if code_email != 0 or not current_email:
        user_email = args.email or input("Enter your Git/GitHub Email: ").strip() or "developer@echopulse.local"
        run_cmd(["git", "config", "user.email", user_email])
        print(f"✓ Configured git user.email: {user_email}")
    else:
        print(f"✓ Git user.email: {current_email}")

    # 3. Initialize Git if needed
    if not (project_dir / ".git").exists():
        print("Initializing new Git repository...")
        run_cmd(["git", "init"])
        print("✓ Git repository initialized.")
    else:
        print("✓ Existing Git repository detected.")

    # 4. Stage and commit files
    print("\nStaging files...")
    run_cmd(["git", "add", "."])

    # Check status
    _, status_out = run_cmd(["git", "status", "--porcelain"], check=False)
    if status_out:
        print("Creating commit...")
        code, commit_out = run_cmd([
            "git", "commit", "-m",
            "Initial commit: EchoPulse AI Immediate Audio Pattern Recognition & Spectral Analysis Engine"
        ], check=False)
        print(f"✓ {commit_out.splitlines()[0] if commit_out else 'Committed changes.'}")
    else:
        print("✓ All files are already committed.")

    # 5. Set default branch to main
    run_cmd(["git", "branch", "-M", args.branch], check=False)

    # 6. Remote Configuration
    repo_url = args.repo
    if not repo_url:
        print("\n" + "-" * 60)
        print("To upload this project to GitHub:")
        print("1. Go to https://github.com/new and create a new repository")
        print("   (e.g., 'echopulse-audio' or 'ai-and-audio').")
        print("2. Do NOT initialize with a README, .gitignore, or license.")
        print("3. Copy the repository HTTPS or SSH URL.")
        print("-" * 60)
        repo_url = input("\nEnter your GitHub repository URL: ").strip()

    if not repo_url:
        print("\n[INFO] No repository URL provided.")
        print("Your project is fully committed locally! When ready, push anytime with:")
        print(f"  git remote add origin <your-repo-url>")
        print(f"  git push -u origin {args.branch}")
        return

    # Check if remote origin already exists
    code, remotes = run_cmd(["git", "remote"], check=False)
    if "origin" in remotes.splitlines():
        run_cmd(["git", "remote", "set-url", "origin", repo_url])
        print(f"✓ Updated remote 'origin' -> {repo_url}")
    else:
        run_cmd(["git", "remote", "add", "origin", repo_url])
        print(f"✓ Added remote 'origin' -> {repo_url}")

    # 7. Push to GitHub
    print(f"\nPushing to GitHub ({args.branch} branch)...")
    code, push_out = run_cmd(["git", "push", "-u", "origin", args.branch], check=False)
    if code == 0:
        print("\n🎉 SUCCESS! Project uploaded to GitHub successfully!")
        print(f"Repository URL: {repo_url}")
    else:
        print(f"\n[Push Result / Output]:\n{push_out}")
        print("\nIf authentication was requested, Git Credential Manager or your SSH key can complete the push:")
        print(f"  git push -u origin {args.branch}")


if __name__ == "__main__":
    main()

