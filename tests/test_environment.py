"""
tests/test_environment.py

Deterministic unit tests for:
  - reset() returns correct observation shape
  - step() produces correct rewards for each task
  - Grader scores pass/fail thresholds correctly
  - Reward function gives partial credit (not just binary)
  - Penalties fire on bad behaviour

Run with:
    pytest tests/ -v
"""

import pytest
from server.models import EmailAction, EmailObservation, EmailState, UrgencyLabel, Department
from server.environment import EmailTriageEnvironment
from server.data import TASKS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return EmailTriageEnvironment()


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_easy_returns_single_email(self, env):
        obs = env.reset(task_id="easy", seed=42)
        assert isinstance(obs, EmailObservation)
        assert obs.single_email is not None
        assert obs.inbox == []
        assert obs.task_id == "easy"
        assert obs.done is False
        assert obs.step_count == 0

    def test_medium_returns_inbox_of_10(self, env):
        obs = env.reset(task_id="medium", seed=42)
        assert len(obs.inbox) == 10
        assert obs.single_email is None

    def test_hard_returns_urgent_email(self, env):
        obs = env.reset(task_id="hard", seed=42)
        gt = env._ground_truths[obs.single_email.id]
        assert gt["urgency"] == UrgencyLabel.URGENT

    def test_state_initialised(self, env):
        env.reset(task_id="easy", seed=1)
        assert env.state.step_count == 0
        assert env.state.done is False
        assert env.state.episode_id is not None

    def test_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="impossible")

    def test_seed_is_reproducible(self, env):
        obs1 = env.reset(task_id="easy", seed=99)
        env2 = EmailTriageEnvironment()
        obs2 = env2.reset(task_id="easy", seed=99)
        assert obs1.single_email.id == obs2.single_email.id
        assert obs1.single_email.subject == obs2.single_email.subject


# ── Easy task ─────────────────────────────────────────────────────────────────

class TestEasyTask:
    def _perfect_action(self, env, obs):
        e  = obs.single_email
        gt = env._ground_truths[e.id]
        return EmailAction(
            action_type="classify",
            email_id=e.id,
            urgency=gt["urgency"],
            priority=gt["priority"],
        )

    def test_perfect_score_close_to_1(self, env):
        obs = env.reset(task_id="easy", seed=7)
        action = self._perfect_action(env, obs)
        obs2 = env.step(action)
        assert obs2.reward >= 0.80
        assert obs2.done is True

    def test_wrong_urgency_lowers_reward(self, env):
        obs = env.reset(task_id="easy", seed=7)
        e   = obs.single_email
        gt  = env._ground_truths[e.id]
        wrong_urgency = next(u for u in UrgencyLabel if u != gt["urgency"])
        action = EmailAction(
            action_type="classify",
            email_id=e.id,
            urgency=wrong_urgency,
            priority=gt["priority"],
        )
        obs2 = env.step(action)
        assert obs2.reward < 0.80
        assert obs2.done is True

    def test_priority_off_by_1_gets_partial_credit(self, env):
        for seed in range(20):
            obs = env.reset(task_id="easy", seed=seed)
            e   = obs.single_email
            gt  = env._ground_truths[e.id]
            if gt["priority"] not in (1, 5):
                adj_priority = gt["priority"] + 1
                action = EmailAction(
                    action_type="classify",
                    email_id=e.id,
                    urgency=gt["urgency"],
                    priority=adj_priority,
                )
                obs2 = env.step(action)
                assert obs2.reward > 0.30
                break

    def test_wrong_action_type_penalised(self, env):
        obs = env.reset(task_id="easy", seed=0)
        action = EmailAction(action_type="rank", ranked_ids=[obs.single_email.id])
        obs2 = env.step(action)
        assert obs2.reward < 0
        assert obs2.done is False

    def test_missing_fields_penalised(self, env):
        obs = env.reset(task_id="easy", seed=0)
        action = EmailAction(action_type="classify", email_id=obs.single_email.id)
        obs2 = env.step(action)
        assert obs2.reward < 0

    def test_episode_ends_after_classify(self, env):
        obs = env.reset(task_id="easy", seed=0)
        action = self._perfect_action(env, obs)
        obs2 = env.step(action)
        assert obs2.done is True
        assert env.state.done is True

    def test_step_after_done_penalised(self, env):
        obs = env.reset(task_id="easy", seed=0)
        obs = env.step(self._perfect_action(env, obs))
        obs2 = env.step(EmailAction(action_type="done"))
        assert obs2.reward < 0
        assert obs2.done is True


# ── Medium task ───────────────────────────────────────────────────────────────

class TestMediumTask:
    def _correct_ranking(self, env, obs):
        ranked = sorted(
            obs.inbox,
            key=lambda e: env._ground_truths[e.id]["priority"],
            reverse=True,
        )
        return EmailAction(
            action_type="rank",
            ranked_ids=[e.id for e in ranked],
        )

    def test_perfect_ranking_gives_high_score(self, env):
        obs = env.reset(task_id="medium", seed=5)
        action = self._correct_ranking(env, obs)
        obs2 = env.step(action)
        assert obs2.reward >= 0.80
        assert obs2.done is True

    def test_worst_ranking_scores_below_perfect(self, env):
        obs = env.reset(task_id="medium", seed=5)
        worst = sorted(obs.inbox, key=lambda e: env._ground_truths[e.id]["priority"])
        action = EmailAction(action_type="rank", ranked_ids=[e.id for e in worst])
        obs_worst = env.step(action)

        env2 = EmailTriageEnvironment()
        obs2 = env2.reset(task_id="medium", seed=5)
        perfect = sorted(obs2.inbox, key=lambda e: env2._ground_truths[e.id]["priority"], reverse=True)
        obs_perfect = env2.step(EmailAction(action_type="rank", ranked_ids=[e.id for e in perfect]))
        assert obs_perfect.reward > obs_worst.reward

    def test_missing_email_id_penalised(self, env):
        obs = env.reset(task_id="medium", seed=5)
        partial = [e.id for e in obs.inbox[:5]]
        action = EmailAction(action_type="rank", ranked_ids=partial)
        obs2 = env.step(action)
        assert obs2.reward < 0

    def test_can_improve_over_multiple_steps(self, env):
        obs = env.reset(task_id="medium", seed=5)
        action1 = EmailAction(action_type="rank", ranked_ids=[e.id for e in obs.inbox])
        obs1    = env.step(action1)

        if obs1.done:
            return

        obs_temp = env.reset(task_id="medium", seed=5)
        ranked   = sorted(obs_temp.inbox, key=lambda e: env._ground_truths[e.id]["priority"], reverse=True)
        action2  = EmailAction(action_type="rank", ranked_ids=[e.id for e in ranked])
        obs2     = env.step(action2)
        assert obs2.correctness_score >= obs1.correctness_score


# ── Hard task ─────────────────────────────────────────────────────────────────

class TestHardTask:
    def _full_action(self, env, obs, reply="", route=None):
        e  = obs.single_email
        gt = env._ground_truths[e.id]
        return EmailAction(
            action_type="triage",
            email_id=e.id,
            urgency=gt["urgency"],
            priority=gt["priority"],
            reply_draft=reply or (
                "Dear Customer,\n\nThank you for reaching out. We have received your message "
                "and are treating this as a priority matter. Our team is investigating and will "
                "respond with a full resolution within 4 hours. We sincerely apologize for any "
                "inconvenience caused.\n\nBest regards,\nSupport Team"
            ),
            route_to=route or gt["department"],
        )

    def test_correct_triage_scores_above_half(self, env):
        obs = env.reset(task_id="hard", seed=3)
        action = self._full_action(env, obs)
        obs2 = env.step(action)
        assert obs2.reward >= 0.50
        assert obs2.done is True

    def test_wrong_route_reduces_score(self, env):
        obs = env.reset(task_id="hard", seed=3)
        gt  = env._ground_truths[obs.single_email.id]
        wrong_dept = next(d for d in Department if d != gt["department"])
        action = self._full_action(env, obs, route=wrong_dept)
        obs2 = env.step(action)
        assert obs2.reward < 0.80

    def test_empty_reply_lowers_score(self, env):
        obs = env.reset(task_id="hard", seed=3)
        action_short = self._full_action(env, obs, reply="ok")
        obs2 = env.step(action_short)

        env2 = EmailTriageEnvironment()
        obs_full = env2.reset(task_id="hard", seed=3)
        obs_full2 = env2.step(self._full_action(env2, obs_full))
        assert obs2.reward < obs_full2.reward

    def test_missing_fields_penalised(self, env):
        obs = env.reset(task_id="hard", seed=3)
        action = EmailAction(
            action_type="triage",
            email_id=obs.single_email.id,
            urgency="urgent",
            priority=5,
        )
        obs2 = env.step(action)
        assert obs2.reward < 0


# ── Reward invariants ─────────────────────────────────────────────────────────

class TestRewardInvariants:
    def test_done_action_immediately_penalised(self, env):
        env.reset(task_id="easy", seed=0)
        obs2 = env.step(EmailAction(action_type="done"))
        assert obs2.reward <= -0.10

    def test_max_steps_exceeded_penalised(self, env):
        obs = env.reset(task_id="easy", seed=0)
        for _ in range(TASKS["easy"]["max_steps"] + 1):
            action = EmailAction(action_type="rank", ranked_ids=[obs.single_email.id])
            obs = env.step(action)
            if obs.done:
                break
        assert obs.reward <= 0

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_state_step_count_increments(self, env, task_id):
        env.reset(task_id=task_id, seed=0)
        assert env.state.step_count == 0

        if task_id == "easy":
            env.step(EmailAction(action_type="done"))
        elif task_id == "medium":
            env.step(EmailAction(action_type="rank", ranked_ids=[e.id for e in env._emails]))
        else:
            env.step(EmailAction(action_type="done"))

        assert env.state.step_count == 1

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_reset_clears_state(self, env, task_id):
        env.reset(task_id=task_id, seed=0)
        env.step(EmailAction(action_type="done"))
        obs = env.reset(task_id=task_id, seed=1)
        assert obs.step_count == 0
        assert obs.cumulative_reward == 0.0
        assert env.state.step_count == 0
        assert env.state.done is False


# ── Grader thresholds ─────────────────────────────────────────────────────────

class TestGraderThresholds:
    @pytest.mark.parametrize("task_id,score,expected_grade", [
        ("easy",   0.95, "excellent"),
        ("easy",   0.75, "pass"),
        ("easy",   0.50, "fail"),
        ("medium", 0.90, "excellent"),
        ("medium", 0.65, "pass"),
        ("medium", 0.40, "fail"),
        ("hard",   0.85, "excellent"),
        ("hard",   0.60, "pass"),
        ("hard",   0.30, "fail"),
    ])
    def test_grade_thresholds(self, task_id, score, expected_grade):
        t = TASKS[task_id]
        if score >= t["excellent_threshold"]:
            grade = "excellent"
        elif score >= t["pass_threshold"]:
            grade = "pass"
        else:
            grade = "fail"
        assert grade == expected_grade
