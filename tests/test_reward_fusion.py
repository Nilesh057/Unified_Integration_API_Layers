import pytest
from core.reward_fusion import fuse_rewards, safe_normalize


class TestSafeNormalize:
    """Test safe_normalize function edge cases."""

    def test_normal_range(self):
        """Test normal min-max normalization."""
        assert safe_normalize(0.5, 0.0, 1.0) == 0.5
        assert safe_normalize(0.0, 0.0, 1.0) == 0.0
        assert safe_normalize(1.0, 0.0, 1.0) == 1.0

    def test_zero_range(self):
        """Test zero range case returns equal weights."""
        assert safe_normalize(5.0, 5.0, 5.0) == 0.5
        assert safe_normalize(10.0, 10.0, 10.0) == 0.5
        assert safe_normalize(0.0, 0.0, 0.0) == 0.5

    def test_clamping(self):
        """Test input clamping to range."""
        assert safe_normalize(-2.0, -1.0, 1.0) == 0.0  # clamped to min
        assert safe_normalize(2.0, -1.0, 1.0) == 1.0   # clamped to max

    def test_different_ranges(self):
        """Test with different min/max ranges."""
        assert safe_normalize(5.0, 0.0, 10.0) == 0.5
        assert safe_normalize(0.0, 0.0, 10.0) == 0.0
        assert safe_normalize(10.0, 0.0, 10.0) == 1.0


class TestRewardFusion:
    """Test reward fusion logic."""

    def test_zero_range_signals(self):
        """Test fusion with zero-range signals."""
        registry = {
            "agents": {
                "rl_agent": {"weight": 0.5, "endpoint": "/api/test1"},
                "summarizer": {"weight": 0.5, "endpoint": "/api/test2"}
            }
        }

        # All signals have same value (zero range)
        result = fuse_rewards(
            rl_reward=0.5,
            user_feedback=0.5,
            action_success=0.5,
            cognitive_score=0.5,
            registry=registry
        )

        # With inputs of 0.5 in range -1 to 1, normalize(0.5) = (0.5 - (-1)) / (1 - (-1)) = 1.5/2 = 0.75
        # So final score = 0.75 * 0.5 + 0.75 * 0.5 = 0.75
        assert result["final_score"] == pytest.approx(0.75, abs=0.01)
        assert result["final_confidence"] >= 0.0
        assert result["final_confidence"] <= 1.0

    def test_negative_confidences(self):
        """Test handling of negative confidence values."""
        registry = {
            "agents": {
                "agent1": {"weight": 0.6, "endpoint": "/api/test1"},
                "agent2": {"weight": 0.4, "endpoint": "/api/test2"}
            }
        }

        result = fuse_rewards(
            rl_reward=0.8,
            user_feedback=0.7,
            action_success=0.6,
            cognitive_score=0.5,
            registry=registry,
            dynamic_confidences={"agent1": -0.5, "agent2": 0.8}  # negative confidence
        )

        # Should handle negative confidence gracefully
        assert result["final_score"] >= 0.0
        assert result["final_score"] <= 1.0
        assert result["final_confidence"] >= 0.0
        assert result["final_confidence"] <= 1.0

    def test_weight_sum_normalization(self):
        """Test that weights are properly normalized to sum to 1."""
        registry = {
            "agents": {
                "rl_agent": {"weight": 0.3, "endpoint": "/api/test1"},
                "summarizer": {"weight": 0.7, "endpoint": "/api/test2"}
            }
        }

        result = fuse_rewards(
            rl_reward=0.6,
            user_feedback=0.4,
            action_success=0.8,
            cognitive_score=0.2,
            registry=registry
        )

        # Check that decision trace contains proper weight information
        total_weight = sum(entry["final_weight"] for entry in result["decision_trace"])
        assert total_weight == pytest.approx(1.0, abs=1e-10)  # Very tight tolerance

        # Verify that weights form a valid probability distribution
        weights = [entry["final_weight"] for entry in result["decision_trace"]]
        assert all(w >= 0.0 for w in weights)  # All weights non-negative
        assert all(w <= 1.0 for w in weights)  # All weights <= 1.0
        assert sum(weights) == pytest.approx(1.0, abs=1e-10)  # Sum to exactly 1.0

    def test_empty_registry(self):
        """Test behavior with empty registry."""
        result = fuse_rewards(
            rl_reward=0.5,
            user_feedback=0.5,
            action_success=0.5,
            cognitive_score=0.5,
            registry={}
        )

        assert result["final_score"] == 0.0
        assert result["final_confidence"] == 0.0
        assert result["top_agent"] == "none"

    def test_extreme_values(self):
        """Test with extreme signal values."""
        registry = {
            "agents": {
                "agent1": {"weight": 1.0, "endpoint": "/api/test1"}
            }
        }

        result = fuse_rewards(
            rl_reward=100.0,  # Should be clamped
            user_feedback=-50.0,  # Should be clamped
            action_success=1.5,  # Should be clamped
            cognitive_score=-2.0,  # Should be clamped
            registry=registry
        )

        assert result["final_score"] >= 0.0
        assert result["final_score"] <= 1.0
        assert result["final_confidence"] >= 0.0
        assert result["final_confidence"] <= 1.0

    def test_entropy_calculation_edge_cases(self):
        """Test entropy calculation with edge cases."""
        registry = {
            "agents": {
                "agent1": {"weight": 1.0, "endpoint": "/api/test1"}
            }
        }

        # Single agent should have zero entropy (perfect confidence)
        result = fuse_rewards(
            rl_reward=0.5,
            user_feedback=0.5,
            action_success=0.5,
            cognitive_score=0.5,
            registry=registry
        )

        assert result["final_confidence"] == 1.0  # Perfect confidence with single agent