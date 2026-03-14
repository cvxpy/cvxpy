"""
Ensue coordinator for solo CVXPY canonicalization research.

Simplified from mutable-state-inc/autoresearch-at-home/coordinator.py.
Provides persistent memory across Claude Code sessions for tracking
experiments, insights, hypotheses, and the current best configuration.

Usage:
    from coordinator import Coordinator
    coord = Coordinator()
    coord.publish_result("description", benchmark_json, git_diff, "keep")
    coord.post_insight("COO backend 2x faster for DPP", ["result_abc123"])
    coord.analyze()
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

ENSUE_BASE_URL = "https://api.ensue-network.ai"
NAMESPACE = "cvxpy-canon-opt"


def _load_api_key() -> str:
    """Load API key from env or file."""
    key = os.environ.get("ENSUE_API_KEY", "")
    if not key:
        key_file = Path(__file__).parent / ".autoresearch-key"
        if key_file.exists():
            key = key_file.read_text().strip()
    return key


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text[:80]


def _experiment_hash(description: str) -> str:
    """Short hash for experiment deduplication."""
    return hashlib.sha256(description.encode()).hexdigest()[:12]


class Coordinator:
    """Solo research coordinator backed by Ensue shared memory."""

    def __init__(self):
        self.api_key = _load_api_key()
        self._connected = None
        self._rpc_id = 0

    @property
    def connected(self) -> bool:
        """Test connectivity to Ensue."""
        if self._connected is None:
            try:
                self._call_tool("list_keys", {"prefix": f"{NAMESPACE}/best/%"})
                self._connected = True
            except Exception:
                self._connected = False
        return self._connected

    def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Make a JSON-RPC 2.0 call to Ensue."""
        if not self.api_key:
            raise RuntimeError(
                "No Ensue API key found. Set ENSUE_API_KEY or create .autoresearch-key"
            )

        self._rpc_id += 1
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": self._rpc_id,
        }

        resp = requests.post(
            ENSUE_BASE_URL,
            headers=headers,
            json=payload,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ensue error {resp.status_code}: {resp.text[:200]}")

        # Response may be SSE format (data: {...}) or plain JSON
        text = resp.text.strip()
        if text.startswith("data: "):
            text = text[len("data: "):]
        data = json.loads(text)

        if "error" in data:
            raise RuntimeError(f"Ensue error: {data['error']}")
        return data.get("result")

    def _structured(self, result: Any) -> dict:
        """Extract structuredContent from an Ensue RPC result."""
        if isinstance(result, dict):
            return result.get("structuredContent") or result
        return {}

    def _set_memory(self, key: str, value: str, embed: bool = False) -> Any:
        """Create or update a memory item."""
        # Try update first, fall back to create
        try:
            return self._call_tool("update_memory", {
                "key_name": key,
                "value": value,
            })
        except Exception:
            return self._call_tool("create_memory", {
                "items": [{
                    "key_name": key,
                    "description": key,
                    "value": value,
                    "embed": embed,
                }],
            })

    def _get_memory(self, key: str) -> str | None:
        """Get a memory item value by key."""
        result = self._call_tool("get_memory", {"key_names": [key]})
        sc = self._structured(result)
        results = sc.get("results", [])
        if results and results[0].get("status") == "success":
            return results[0].get("value")
        return None

    def _list_keys(self, prefix: str) -> list[str]:
        """List keys with a given prefix (use SQL LIKE % wildcard)."""
        result = self._call_tool("list_keys", {"prefix": f"{prefix}%"})
        sc = self._structured(result)
        return [k["key_name"] for k in sc.get("keys", []) if "key_name" in k]

    def _search(self, query: str, limit: int = 10, prefix: str = "") -> list[dict]:
        """Semantic search over memories."""
        args = {"query": query, "limit": limit}
        if prefix:
            args["prefix"] = prefix
        result = self._call_tool("search_memories", args)
        sc = self._structured(result)
        return sc.get("results", [])

    # ── Results ──────────────────────────────────────────────────────

    def publish_result(
        self,
        description: str,
        benchmark_json: dict | str,
        git_diff: str = "",
        status: str = "keep",
    ) -> str:
        """
        Publish an experiment result.

        Args:
            description: What was changed/tested
            benchmark_json: Benchmark output (dict or JSON string)
            git_diff: The code diff for this experiment
            status: "keep" (promising), "discard" (worse), or "error"

        Returns:
            The result key.
        """
        if isinstance(benchmark_json, str):
            benchmark_json = json.loads(benchmark_json)

        exp_hash = _experiment_hash(description)
        key = f"{NAMESPACE}/results/{exp_hash}"

        value = {
            "description": description,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "benchmark": benchmark_json,
            "git_diff": git_diff,
        }

        self._set_memory(key, json.dumps(value), embed=True)

        # Update best if this is better
        if status == "keep" and "geomean_ms" in benchmark_json:
            self._maybe_update_best(description, benchmark_json, git_diff)

        print(f"Published result: {description} [{status}] -> {key}")
        return key

    def _maybe_update_best(self, description: str, benchmark: dict, git_diff: str):
        """Update best/ if this result beats the current best."""
        new_score = benchmark.get("geomean_ms", float("inf"))

        try:
            current = self._get_memory(f"{NAMESPACE}/best/metadata")
            if current:
                current_data = json.loads(current)
                if current_data.get("geomean_ms", float("inf")) <= new_score:
                    return  # Current best is still better
        except Exception:
            pass  # No current best, this becomes the best

        metadata = {
            "description": description,
            "geomean_ms": new_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark": benchmark,
        }
        self._set_memory(f"{NAMESPACE}/best/metadata", json.dumps(metadata))
        if git_diff:
            self._set_memory(f"{NAMESPACE}/best/config", git_diff)
        print(f"  ** New best! geomean={new_score:.2f}ms")

    def list_results(self) -> list[dict]:
        """List all published results."""
        keys = self._list_keys(f"{NAMESPACE}/results/")
        results = []
        for key in keys:
            try:
                val = self._get_memory(key)
                if val:
                    results.append(json.loads(val))
            except Exception:
                continue
        return results

    # ── Insights ─────────────────────────────────────────────────────

    def post_insight(self, text: str, evidence_keys: list[str] | None = None) -> str:
        """
        Post a research insight.

        Args:
            text: The insight description
            evidence_keys: Optional list of result keys supporting this insight

        Returns:
            The insight key.
        """
        slug = _slugify(text)
        key = f"{NAMESPACE}/insights/{slug}"

        value = {
            "text": text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence": evidence_keys or [],
        }
        self._set_memory(key, json.dumps(value), embed=True)
        print(f"Posted insight: {text[:60]}...")
        return key

    def list_insights(self) -> list[dict]:
        """List all insights."""
        keys = self._list_keys(f"{NAMESPACE}/insights/")
        insights = []
        for key in keys:
            try:
                val = self._get_memory(key)
                if val:
                    insights.append(json.loads(val))
            except Exception:
                continue
        return insights

    # ── Hypotheses ───────────────────────────────────────────────────

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        priority: int = 2,
        status: str = "open",
    ) -> str:
        """
        Publish a research hypothesis to try.

        Args:
            title: Short title
            hypothesis: Detailed description of what to try
            priority: 1 (high) to 3 (low)
            status: "open", "in_progress", "tested", "rejected"

        Returns:
            The hypothesis key.
        """
        slug = _slugify(title)
        key = f"{NAMESPACE}/hypotheses/{slug}"

        value = {
            "title": title,
            "hypothesis": hypothesis,
            "priority": priority,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._set_memory(key, json.dumps(value), embed=True)
        print(f"Published hypothesis [P{priority}]: {title}")
        return key

    def list_hypotheses(self, status: str | None = None) -> list[dict]:
        """List hypotheses, optionally filtered by status."""
        keys = self._list_keys(f"{NAMESPACE}/hypotheses/")
        hypotheses = []
        for key in keys:
            try:
                val = self._get_memory(key)
                if val:
                    h = json.loads(val)
                    if status is None or h.get("status") == status:
                        hypotheses.append(h)
            except Exception:
                continue
        # Sort by priority
        hypotheses.sort(key=lambda h: h.get("priority", 99))
        return hypotheses

    # ── Best Config ──────────────────────────────────────────────────

    def pull_best(self) -> dict:
        """Get the current best configuration."""
        try:
            val = self._get_memory(f"{NAMESPACE}/best/metadata")
            if val:
                meta = json.loads(val)
                diff_val = self._get_memory(f"{NAMESPACE}/best/config")
                meta["git_diff"] = diff_val or ""
                return meta
        except Exception:
            pass
        return {"description": "No best yet", "geomean_ms": None}

    # ── Analysis ─────────────────────────────────────────────────────

    def analyze(self) -> str:
        """Print a summary analysis of all research so far."""
        results = self.list_results()
        insights = self.list_insights()
        hypotheses = self.list_hypotheses()
        best = self.pull_best()

        lines = []
        lines.append("=" * 60)
        lines.append("CVXPY Canonicalization Research Summary")
        lines.append("=" * 60)

        # Best
        lines.append(f"\nBest config: {best.get('description', 'none')}")
        if best.get("geomean_ms"):
            lines.append(f"  geomean: {best['geomean_ms']:.2f}ms")

        # Results
        lines.append(f"\nResults: {len(results)} experiments")
        keep = [r for r in results if r.get("status") == "keep"]
        discard = [r for r in results if r.get("status") == "discard"]
        errors = [r for r in results if r.get("status") == "error"]
        lines.append(f"  keep={len(keep)}, discard={len(discard)}, error={len(errors)}")

        for r in sorted(results, key=lambda x: x.get("timestamp", ""))[-5:]:
            geomean = r.get("benchmark", {}).get("geomean_ms", "?")
            lines.append(
                f"  [{r.get('status', '?')}] {r.get('description', '?')[:50]} "
                f"(geomean={geomean})"
            )

        # Insights
        lines.append(f"\nInsights: {len(insights)}")
        for i in insights[-5:]:
            lines.append(f"  - {i.get('text', '?')[:70]}")

        # Hypotheses
        open_h = [h for h in hypotheses if h.get("status") == "open"]
        lines.append(f"\nHypotheses: {len(hypotheses)} total, {len(open_h)} open")
        for h in open_h[:5]:
            lines.append(f"  [P{h.get('priority', '?')}] {h.get('title', '?')}")

        output = "\n".join(lines)
        print(output)
        return output

    def ask(self, query: str) -> list[dict]:
        """
        Semantic search over results and insights.

        Args:
            query: Natural language query

        Returns:
            List of matching items.
        """
        matches = self._search(query, prefix=f"{NAMESPACE}/")
        if matches:
            print(f"Found {len(matches)} matches for '{query}':")
            for m in matches[:10]:
                if isinstance(m, dict):
                    key = m.get("key_name", "")
                    desc = m.get("description", key)
                    print(f"  {desc[:80]}")
                else:
                    print(f"  {str(m)[:80]}")
        else:
            print(f"No matches for '{query}'")
        return matches


if __name__ == "__main__":
    coord = Coordinator()
    print(f"Connected: {coord.connected}")
    if coord.connected:
        coord.analyze()
