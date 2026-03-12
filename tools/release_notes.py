"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Generate the "PR by author" list for CVXPY release notes.

Compares the release tag against the previous release branch so that
cherry-picked patch PRs are not duplicated.

Usage:
    python tools/release_notes.py v1.8.0 release/1.7.x
    python tools/release_notes.py v1.8.0  # auto-detects previous release branch
"""

import json
import re
import subprocess
import sys


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def parse_tag(tag: str) -> tuple[int, int, int]:
    """Parse a tag like v1.8.0 into (major, minor, micro)."""
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if not m:
        raise ValueError(f"Cannot parse tag: {tag}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_previous_release_branch(tag: str) -> str:
    """Infer the previous release branch from a tag like v1.8.0."""
    major, minor, _ = parse_tag(tag)
    return f"origin/release/{major}.{minor - 1}.x"


def get_previous_release_tag(tag: str) -> str:
    """Infer the previous minor release tag from a tag like v1.8.0."""
    major, minor, _ = parse_tag(tag)
    return f"v{major}.{minor - 1}.0"


def extract_pr_numbers(log_output: str) -> set[int]:
    """Extract PR numbers from git log output.

    Matches the (#NNNN) pattern at the end of a line, which is the
    convention for merge and squash commit subject lines. This avoids
    false positives from inline issue references like "Closes #1000".
    """
    prs = set()
    for m in re.finditer(r"\(#(\d+)\)\s*$", log_output, re.MULTILINE):
        prs.add(int(m.group(1)))
    return prs


def get_pr_author(pr_number: int) -> str | None:
    """Look up the GitHub username for a PR."""
    try:
        result = run([
            "gh", "pr", "view", str(pr_number),
            "--json", "author", "-q", ".author.login",
        ])
        return result
    except subprocess.CalledProcessError:
        return None


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tag>")
        print(f"  e.g. {sys.argv[0]} v1.8.0  (minor release)")
        print(f"  e.g. {sys.argv[0]} v1.7.5  (patch release)")
        sys.exit(1)

    tag = sys.argv[1]
    major, minor, micro = parse_tag(tag)

    if micro == 0:
        # Minor release: compare against previous minor's .0 tag,
        # excluding PRs cherry-picked onto the previous release branch.
        prev_tag = get_previous_release_tag(tag)
        prev_branch = get_previous_release_branch(tag)

        release_log = run(
            ["git", "log", "--format=%B", f"{prev_tag}..{tag}"]
        )
        release_prs = extract_pr_numbers(release_log)

        patch_log = run(
            ["git", "log", "--format=%B", f"{prev_tag}..{prev_branch}"]
        )
        patch_prs = extract_pr_numbers(patch_log)

        pr_numbers = sorted(release_prs - patch_prs)

        print(
            f"Found {len(release_prs)} PRs in {prev_tag}..{tag}, "
            f"excluding {len(patch_prs)} patch PRs from "
            f"{prev_branch}, {len(pr_numbers)} remaining.",
            file=sys.stderr,
        )
    else:
        # Patch release: compare against the previous tag on the
        # same release branch.
        prev_tag = f"v{major}.{minor}.{micro - 1}"

        release_log = run(
            ["git", "log", "--format=%B", f"{prev_tag}..{tag}"]
        )
        pr_numbers = sorted(extract_pr_numbers(release_log))

        print(
            f"Found {len(pr_numbers)} PRs in {prev_tag}..{tag}.",
            file=sys.stderr,
        )

    print("Looking up authors...", file=sys.stderr)

    # Look up authors via gh CLI, batch with graphql to reduce API calls
    author_prs: dict[str, list[int]] = {}
    skipped: list[int] = []

    # Use graphql to batch lookups (100 at a time)
    for i in range(0, len(pr_numbers), 100):
        batch = pr_numbers[i:i + 100]
        # Build a graphql query for this batch
        parts = []
        for j, pr in enumerate(batch):
            parts.append(
                f'pr{j}: pullRequest(number: {pr}) {{ author {{ login }} }}'
            )
        query = "{ repository(owner: \"cvxpy\", name: \"cvxpy\") { " + " ".join(parts) + " } }"
        try:
            result = run(["gh", "api", "graphql", "-f", f"query={query}"])
            data = json.loads(result)["data"]["repository"]
            for j, pr in enumerate(batch):
                node = data.get(f"pr{j}")
                if node and node.get("author"):
                    login = node["author"]["login"]
                    if "dependabot" in login:
                        continue
                    author_prs.setdefault(login, []).append(pr)
                else:
                    skipped.append(pr)
        except subprocess.CalledProcessError:
            # Fall back to individual lookups
            for pr in batch:
                author = get_pr_author(pr)
                if author is None:
                    skipped.append(pr)
                elif "dependabot" in author:
                    continue
                else:
                    author_prs.setdefault(author, []).append(pr)

    # Count PRs (excluding dependabot and skipped)
    total_prs = sum(len(prs) for prs in author_prs.values())
    total_authors = len(author_prs)

    print(f"\nThis new release totaled {total_prs} PRs from {total_authors} contributors.\n")
    for author in sorted(author_prs, key=lambda a: a.lower()):
        prs = ", ".join(f"#{pr}" for pr in sorted(author_prs[author]))
        print(f"- @{author} | {prs}")

    if skipped:
        skipped_str = ", ".join(f"#{pr}" for pr in skipped)
        print(f"\nSkipped PRs (not found): {skipped_str}", file=sys.stderr)


if __name__ == "__main__":
    main()
