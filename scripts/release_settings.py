"""Check, or apply, the repository settings the release flow depends on.

Workflow: run by hand by a repository admin. These settings live in GitHub, not
in the tree, so this script is their record; `check` reports drift from it.

- A tag ruleset on `v*`: only the release team creates, moves or deletes a
  release tag.
- The `release` environment (ci-release's publish job): deploys only from `v*`
  tags and waits for a reviewer from the release team.
- The `pre-release` environment (ci-pre-release): deploys only from `release-*`
  branches and waits for the same reviewers.

Self-review stays allowed, so one admin can approve their own release.

    python3 scripts/release_settings.py            # check: GET requests only
    python3 scripts/release_settings.py apply --dry-run
    python3 scripts/release_settings.py apply      # needs repository admin
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


REPO = "xorq-labs/xorq"
# Who may push release tags and approve publishing. Change membership in the
# team, not here.
RELEASE_TEAM = "letsql"
RULESET_NAME = "release-tags"
TAG_RULES = ("creation", "update", "deletion", "non_fast_forward")
# environment name -> the refs it may deploy from, as (type, pattern)
ENVIRONMENTS = {
    "release": {("tag", "v*")},
    "pre-release": {("branch", "release-*")},
}


def get(path: str) -> Any:
    return json.loads(
        subprocess.run(
            ("gh", "api", path), capture_output=True, text=True, check=True
        ).stdout
    )


def send(method: str, path: str, body: dict | None, dry_run: bool) -> None:
    sys.stdout.write(f"{method} {path}\n")
    if body is not None:
        sys.stdout.write(json.dumps(body, indent=2) + "\n")
    if dry_run:
        return
    subprocess.run(
        ("gh", "api", "--method", method, path, "--input", "-"),
        input=json.dumps(body or {}),
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def ruleset_body(team_id: int) -> dict:
    return {
        "name": RULESET_NAME,
        "target": "tag",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
        "rules": [
            {"type": "update", "parameters": {"update_allows_fetch_and_merge": False}}
            if rule == "update"
            else {"type": rule}
            for rule in TAG_RULES
        ],
        "bypass_actors": [
            {"actor_id": team_id, "actor_type": "Team", "bypass_mode": "always"}
        ],
    }


def environment_body(team_id: int) -> dict:
    return {
        "reviewers": [{"type": "Team", "id": team_id}],
        "prevent_self_review": False,
        "deployment_branch_policy": {
            "protected_branches": False,
            "custom_branch_policies": True,
        },
    }


def find_ruleset() -> dict | None:
    for ruleset in get(f"repos/{REPO}/rulesets?per_page=100"):
        if ruleset["name"] == RULESET_NAME:
            return get(f"repos/{REPO}/rulesets/{ruleset['id']}")
    return None


def ruleset_drift(ruleset: dict | None, team_id: int) -> list[str]:
    if ruleset is None:
        return [f"ruleset {RULESET_NAME!r} does not exist"]
    want = ruleset_body(team_id)
    have = {
        "target": ruleset["target"],
        "enforcement": ruleset["enforcement"],
        "include": ruleset["conditions"]["ref_name"]["include"],
        "rules": sorted(rule["type"] for rule in ruleset["rules"]),
        "bypass": [
            (actor["actor_type"], actor["actor_id"], actor["bypass_mode"])
            for actor in ruleset["bypass_actors"]
        ],
    }
    expected = {
        "target": want["target"],
        "enforcement": want["enforcement"],
        "include": want["conditions"]["ref_name"]["include"],
        "rules": sorted(TAG_RULES),
        "bypass": [("Team", team_id, "always")],
    }
    return [
        f"ruleset {RULESET_NAME!r} {key}: have {have[key]}, want {expected[key]}"
        for key in expected
        if have[key] != expected[key]
    ]


def environment_drift(
    name: str, environment: dict, policies: set[tuple[str, str]], team_id: int
) -> list[str]:
    reviewers = [
        (reviewer["type"], reviewer["reviewer"]["id"])
        for rule in environment["protection_rules"]
        if rule["type"] == "required_reviewers"
        for reviewer in rule["reviewers"]
    ]
    self_review = [
        rule["prevent_self_review"]
        for rule in environment["protection_rules"]
        if rule["type"] == "required_reviewers"
    ]
    drift = []
    if reviewers != [("Team", team_id)]:
        drift.append(f"{name}: reviewers {reviewers}, want [('Team', {team_id})]")
    if self_review != [False]:
        drift.append(f"{name}: prevent_self_review {self_review}, want [False]")
    if environment["deployment_branch_policy"] != {
        "protected_branches": False,
        "custom_branch_policies": True,
    }:
        drift.append(
            f"{name}: deployment_branch_policy {environment['deployment_branch_policy']}, want custom policies"
        )
    if policies != ENVIRONMENTS[name]:
        drift.append(
            f"{name}: deploys from {sorted(policies)}, want {sorted(ENVIRONMENTS[name])}"
        )
    return drift


def branch_policies(name: str) -> list[dict]:
    environment = get(f"repos/{REPO}/environments/{name}")
    if environment["deployment_branch_policy"] is None:
        return []
    return get(f"repos/{REPO}/environments/{name}/deployment-branch-policies")[
        "branch_policies"
    ]


def check() -> int:
    team_id = get(f"orgs/{REPO.split('/')[0]}/teams/{RELEASE_TEAM}")["id"]
    drift = ruleset_drift(find_ruleset(), team_id)
    for name in ENVIRONMENTS:
        environment = get(f"repos/{REPO}/environments/{name}")
        policies = {(p["type"], p["name"]) for p in branch_policies(name)}
        drift += environment_drift(name, environment, policies, team_id)
    for line in drift:
        sys.stdout.write(f"drift: {line}\n")
    if not drift:
        sys.stdout.write("release settings match\n")
    return 1 if drift else 0


def apply(dry_run: bool) -> int:
    team_id = get(f"orgs/{REPO.split('/')[0]}/teams/{RELEASE_TEAM}")["id"]
    ruleset = find_ruleset()
    if ruleset is None:
        send("POST", f"repos/{REPO}/rulesets", ruleset_body(team_id), dry_run)
    elif ruleset_drift(ruleset, team_id):
        send(
            "PUT",
            f"repos/{REPO}/rulesets/{ruleset['id']}",
            ruleset_body(team_id),
            dry_run,
        )
    for name, wanted in ENVIRONMENTS.items():
        existing = branch_policies(name)
        send(
            "PUT",
            f"repos/{REPO}/environments/{name}",
            environment_body(team_id),
            dry_run,
        )
        have = {(p["type"], p["name"]) for p in existing}
        for kind, pattern in sorted(wanted - have):
            send(
                "POST",
                f"repos/{REPO}/environments/{name}/deployment-branch-policies",
                {"name": pattern, "type": kind},
                dry_run,
            )
        for policy in existing:
            if (policy["type"], policy["name"]) not in wanted:
                send(
                    "DELETE",
                    f"repos/{REPO}/environments/{name}/deployment-branch-policies/{policy['id']}",
                    None,
                    dry_run,
                )
    return 0 if dry_run else check()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", nargs="?", choices=("check", "apply"), default="check"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="with apply: print the requests only"
    )
    args = parser.parse_args(argv)
    if args.command == "apply":
        return apply(args.dry_run)
    return check()


if __name__ == "__main__":
    sys.exit(main())
