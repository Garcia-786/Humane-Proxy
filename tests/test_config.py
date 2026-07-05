"""Tests for humane_proxy.config (layered config loader)."""

import os

import pytest

from humane_proxy.config import _build_config, _deep_merge, reload_config


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_override(self):
        base = {"safety": {"risk_threshold": 0.7, "spike_boost": 0.25}}
        override = {"safety": {"risk_threshold": 0.5}}
        result = _deep_merge(base, override)
        assert result["safety"]["risk_threshold"] == 0.5
        assert result["safety"]["spike_boost"] == 0.25

    def test_non_mutating(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # original unchanged


class TestLoadConfig:
    def test_defaults_loaded(self):
        config = _build_config()
        assert "safety" in config
        assert "heuristics" in config
        assert config["safety"]["risk_threshold"] == 0.7

    def test_user_config_override(self, tmp_path, monkeypatch):
        user_cfg = tmp_path / "humane_proxy.yaml"
        user_cfg.write_text("safety:\n  risk_threshold: 0.5\n")
        monkeypatch.setenv("HUMANE_PROXY_CONFIG", str(user_cfg))

        config = _build_config()
        assert config["safety"]["risk_threshold"] == 0.5
        # Defaults still present for other keys.
        assert config["safety"]["spike_boost"] == 0.25

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("HUMANE_PROXY_PORT", "9999")
        config = _build_config()
        assert config["server"]["port"] == 9999

    def test_reload_picks_up_changes(self, tmp_path, monkeypatch):
        user_cfg = tmp_path / "humane_proxy.yaml"
        user_cfg.write_text("safety:\n  risk_threshold: 0.6\n")
        monkeypatch.setenv("HUMANE_PROXY_CONFIG", str(user_cfg))

        config1 = reload_config()
        assert config1["safety"]["risk_threshold"] == 0.6

        user_cfg.write_text("safety:\n  risk_threshold: 0.3\n")
        config2 = reload_config()
        assert config2["safety"]["risk_threshold"] == 0.3

    def test_self_harm_keywords_in_defaults(self):
        config = _build_config()
        keywords = config.get("heuristics", {}).get("self_harm_keywords", [])
        assert "suicide" in keywords
        assert "kill myself" in keywords

    def test_criminal_keywords_in_defaults(self):
        config = _build_config()
        keywords = config.get("heuristics", {}).get("criminal_keywords", [])
        assert "how to make a bomb" in keywords

    def test_context_reducers_in_defaults(self):
        config = _build_config()
        reducers = config.get("heuristics", {}).get("context_reducers", [])
        assert "laughing" in reducers
        assert "warning signs" in reducers


class TestMergedConfigReachesClassifiers:
    """Regression: heuristics and trajectory used the legacy package-only
    load_config() at import time, so user humane_proxy.yaml overrides and
    documented env vars (e.g. HUMANE_PROXY_DECAY_HALF_LIFE) were ignored."""

    def test_decay_half_life_env_var_honored(self, monkeypatch):
        monkeypatch.setenv("HUMANE_PROXY_DECAY_HALF_LIFE", "6")
        from humane_proxy.config import reload_config
        reload_config()

        from humane_proxy.risk import trajectory as traj
        traj.detect_spike("cfg-decay-sess", 0.1)  # triggers refresh
        assert traj._DECAY_HALF_LIFE_HOURS == 6.0

        monkeypatch.delenv("HUMANE_PROXY_DECAY_HALF_LIFE")
        reload_config()
        traj.detect_spike("cfg-decay-sess-2", 0.1)
        assert traj._DECAY_HALF_LIFE_HOURS == 24.0

    def test_user_yaml_trajectory_override_honored(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "humane_proxy.yaml"
        cfg_file.write_text("trajectory:\n  spike_delta: 0.9\n", encoding="utf-8")
        monkeypatch.setenv("HUMANE_PROXY_CONFIG", str(cfg_file))

        from humane_proxy.config import reload_config
        reload_config()

        from humane_proxy.risk import trajectory as traj
        traj.detect_spike("cfg-yaml-sess", 0.1)
        assert traj._SPIKE_DELTA == 0.9

        monkeypatch.delenv("HUMANE_PROXY_CONFIG")
        reload_config()
        traj.detect_spike("cfg-yaml-sess-2", 0.1)
        assert traj._SPIKE_DELTA == 0.35

    def test_user_yaml_heuristics_keywords_honored(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "humane_proxy.yaml"
        cfg_file.write_text(
            "heuristics:\n"
            "  self_harm_keywords:\n"
            "    - 'zzz custom marker phrase'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HUMANE_PROXY_CONFIG", str(cfg_file))

        from humane_proxy.config import reload_config
        reload_config()

        from humane_proxy.classifiers.heuristics import classify
        category, score, triggers = classify("zzz custom marker phrase")
        assert category == "self_harm"
        assert any("zzz custom marker phrase" in t for t in triggers)

        monkeypatch.delenv("HUMANE_PROXY_CONFIG")
        reload_config()
        category, _, _ = classify("zzz custom marker phrase")
        assert category == "safe"

    def test_no_duplicate_top_level_keys_in_package_defaults(self):
        """PyYAML silently keeps only the last duplicate key — guard against
        reintroducing a second safety:/escalation: block."""
        import re
        from pathlib import Path
        import humane_proxy

        text = (Path(humane_proxy.__file__).parent / "config.yaml").read_text(encoding="utf-8")
        top_keys = re.findall(r"^([A-Za-z_][A-Za-z0-9_]*):", text, re.M)
        assert len(top_keys) == len(set(top_keys)), f"duplicate top-level keys: {top_keys}"
