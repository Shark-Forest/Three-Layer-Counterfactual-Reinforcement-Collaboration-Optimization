import math
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src28.config import *


class OnlineAnswerVerifier:
    """
    轻量在线 verifier。

    当前版本使用同一个小 embedding backbone + 一个线性 scorer 头。

    这个 scorer 统一回答一个问题：
    - 一条 answer utterance 本身有多值得保留

    因而：
    - keep_prob = scorer(question, incumbent_utterance)
    - accept_prob = scorer(question, candidate_utterance)

    决策规则统一成：
    1. phase 看 incumbent 的绝对分数是否足够高
    2. accept 看 candidate 的分数是否高于 incumbent

    这里不再使用 last_comment。
    verifier 只看“题目 + 产生答案的那段完整文本”，
    因为在 MAS 里，答案自己的 utterance 才是最直接的 supporting source。
    """

    def __init__(self):
        self.lr = VERIFIER_LR
        self.device = "cpu"
        self.embedding_cache = {}

        model_path = VERIFIER_EMBED_MODEL_PATH
        model_source = model_path if os.path.exists(model_path) else VERIFIER_EMBED_MODEL_ID
        load_kwargs = {
            "cache_dir": MODEL_CACHE,
        }
        if model_source == model_path:
            load_kwargs["local_files_only"] = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                **load_kwargs,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModel.from_pretrained(
                model_source,
                **load_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "加载 verifier embedding 模型失败。"
                f" 已尝试 source={model_source!r}。"
                f" 如需离线运行，请预先把模型放到 {model_path}，"
                " 或通过 MAS_VERIFIER_EMBED_MODEL / MAS_VERIFIER_EMBED_MODEL_PATH 覆盖来源。"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "n_embd", None)
        if hidden_size is None:
            raise RuntimeError("无法从 verifier embedding 模型配置中解析 hidden_size。")

        self.score_weights = np.zeros(hidden_size, dtype=np.float64)
        self.score_bias = 0.0

    def _latest_comment(self, history):
        comments = [turn["text"] for turn in history if turn["kind"] == "comment"]
        return comments[-1] if comments else ""

    def _build_score_input_text(self, question, answer_text):
        return (
            f"[QUESTION]\n{question.strip()}\n\n"
            f"[ANSWER_UTTERANCE]\n{(answer_text or '<none>').strip()}\n\n"
            "[TASK]\nEstimate how reliable this answer utterance is."
        )

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked = last_hidden_state * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return masked.sum(dim=1) / denom

    def _encode_text(self, text):
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=VERIFIER_MAX_LENGTH,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self._mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )[0]

        vector = embedding.detach().cpu().numpy().astype(np.float64)
        if len(self.embedding_cache) >= VERIFIER_CACHE_SIZE:
            self.embedding_cache.clear()
        self.embedding_cache[text] = vector
        return vector

    def _sigmoid(self, logit):
        return 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))

    def _safe_logit_from_prob(self, prob):
        clipped = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        return math.log(clipped / (1.0 - clipped))

    def _normalize_features(self, features):
        """
        把 embedding 归一化后再做在线逻辑回归。

        之前直接拿原始 pooled embedding 更新时，向量范数很大，
        一次 SGD 就足以把 logit 推到极端区间，导致 accept 概率很快塌成 0/1。
        """
        feature_norm = float(np.linalg.norm(features))
        if not np.isfinite(feature_norm) or feature_norm < 1e-12:
            return np.zeros_like(features), 0.0
        return features / feature_norm, feature_norm

    def _forward_score(self, question, answer_text):
        if answer_text is None:
            return {
                "prob": 0.0,
                "logit": 0.0,
                "features": None,
                "feature_norm": 0.0,
            }

        text = self._build_score_input_text(question, answer_text)
        raw_features = self._encode_text(text)
        features, feature_norm = self._normalize_features(raw_features)
        logit = float(np.dot(self.score_weights, features) + self.score_bias)
        prob = self._sigmoid(logit)
        return {
            "prob": prob,
            "logit": logit,
            "features": features,
            "feature_norm": feature_norm,
        }

    def _forward_keep(self, question, history, incumbent_text):
        return self._forward_score(question, incumbent_text)

    def _forward_accept(self, question, history, incumbent_text, current_text):
        return self._forward_score(question, current_text)

    def predict_keep_prob(self, question, history, incumbent_text):
        forward = self._forward_keep(
            question,
            history,
            incumbent_text,
        )
        return forward["prob"], forward["features"]

    def predict_accept_prob(self, question, history, incumbent_text, current_text):
        forward = self._forward_accept(
            question,
            history,
            incumbent_text,
            current_text,
        )
        return forward["prob"], forward["features"]

    def get_accept_gate(self, question, history, incumbent_text):
        """
        单 scorer 版本下，accept 屏障就是 incumbent 自身的分数。

        也就是说：
        - keep_prob = scorer(question, incumbent_utterance)
        - accept_prob = scorer(question, candidate_utterance)
        - accept iff accept_prob > keep_prob

        因而这里返回的 barrier 不再包含额外 accept threshold。
        """
        if incumbent_text is None:
            return {
                "threshold": 0.0,
                "keep_prob": 0.0,
                "barrier": 0.0,
            }

        keep_forward = self._forward_keep(
            question,
            history,
            incumbent_text,
        )
        keep_prob = keep_forward["prob"]
        return {
            "threshold": 0.0,
            "keep_prob": keep_prob,
            "barrier": keep_prob,
        }

    def should_accept(
        self,
        question,
        history,
        incumbent_text,
        current_text,
        candidate_pred,
    ):
        if current_text is None or candidate_pred is None:
            barrier = 0.0
            return {
                "accept": False,
                "accept_prob": 0.0,
                "accept_threshold": 0.0,
                "keep_prob": 0.0,
                "accept_barrier": barrier,
                "accept_margin": -barrier,
                "reason": "missing_candidate",
            }
        if incumbent_text is None:
            return {
                "accept": True,
                "accept_prob": 1.0,
                "accept_threshold": 0.0,
                "keep_prob": 0.0,
                "accept_barrier": 0.0,
                "accept_margin": 1.0,
                "reason": "no_incumbent",
            }

        gate = self.get_accept_gate(
            question,
            history,
            incumbent_text,
        )
        barrier = gate["barrier"]
        if current_text == incumbent_text:
            return {
                "accept": False,
                "accept_prob": 0.0,
                "accept_threshold": gate["threshold"],
                "keep_prob": gate["keep_prob"],
                "accept_barrier": barrier,
                "accept_margin": -barrier,
                "reason": "same_text",
            }

        forward = self._forward_accept(
            question,
            history,
            incumbent_text,
            current_text,
        )
        prob = forward["prob"]
        margin = prob - barrier
        return {
            "accept": bool(margin > 0.0),
            "accept_prob": prob,
            "accept_threshold": gate["threshold"],
            "keep_prob": gate["keep_prob"],
            "accept_barrier": barrier,
            "accept_margin": margin,
            "reason": "accept_beats_keep" if margin > 0.0 else "accept_below_keep",
        }

    def update_keep(
        self,
        question,
        history,
        incumbent_text,
        incumbent_reward,
    ):
        if incumbent_text is None:
            return 0.0

        forward = self._forward_keep(
            question,
            history,
            incumbent_text,
        )
        prob = forward["prob"]
        features = forward["features"]
        target = self._reward_to_target_prob(incumbent_reward)
        error = prob - target
        self.score_weights -= self.lr * error * features
        self.score_bias -= self.lr * error
        return prob

    def _reward_to_target_prob(self, reward):
        min_reward = float(REWARD_MISSING_PENALTY)
        max_reward = 1.0
        clipped = min(max(float(reward), min_reward), max_reward)
        scale = max(max_reward - min_reward, 1e-6)
        return (clipped - min_reward) / scale

    def _prepare_score_example(
        self,
        question,
        answer_text,
        reward,
        candidate_pred=None,
        delta_reward=None,
        selected=False,
        role="candidate",
    ):
        if answer_text is None:
            return None

        forward = self._forward_score(question, answer_text)
        return {
            "answer_text": answer_text,
            "candidate_pred": candidate_pred,
            "reward": float(reward),
            "target_prob": self._reward_to_target_prob(reward),
            "delta_reward": None if delta_reward is None else float(delta_reward),
            "selected": bool(selected),
            "role": role,
            "features": forward["features"],
            "feature_norm": forward["feature_norm"],
            "pre_logit": forward["logit"],
            "pre_prob": forward["prob"],
        }

    def _update_score_margin(self, incumbent_example, candidate_example):
        """
        直接学习 candidate 相对 incumbent 的“应有优势”。

        不再把 accept 学成“candidate 自己绝对应该是几分”，
        而是学成：
        - 候选若比 incumbent 好，score(candidate) 应高于 score(incumbent)
        - 候选若比 incumbent 差，score(candidate) 应低于 score(incumbent)
        - 两者差多少，直接由 reward gap 决定

        这样 accept 的训练目标与推理判定 `candidate_score > incumbent_score`
        是同一种机制，不需要再额外引入一个固定 margin 超参。
        """
        incumbent_logit = float(
            np.dot(self.score_weights, incumbent_example["features"]) + self.score_bias
        )
        candidate_logit = float(
            np.dot(self.score_weights, candidate_example["features"]) + self.score_bias
        )
        incumbent_prob = self._sigmoid(incumbent_logit)
        candidate_prob = self._sigmoid(candidate_logit)

        target_margin = candidate_example["target_prob"] - incumbent_example["target_prob"]
        pre_margin = candidate_prob - incumbent_prob
        error = pre_margin - target_margin

        candidate_grad = candidate_prob * (1.0 - candidate_prob) * candidate_example["features"]
        incumbent_grad = incumbent_prob * (1.0 - incumbent_prob) * incumbent_example["features"]
        weight_grad = error * (candidate_grad - incumbent_grad)
        bias_grad = error * (
            candidate_prob * (1.0 - candidate_prob)
            - incumbent_prob * (1.0 - incumbent_prob)
        )

        weight_step = self.lr * weight_grad
        self.score_weights -= weight_step
        self.score_bias -= self.lr * bias_grad

        post_incumbent_logit = float(
            np.dot(self.score_weights, incumbent_example["features"]) + self.score_bias
        )
        post_candidate_logit = float(
            np.dot(self.score_weights, candidate_example["features"]) + self.score_bias
        )
        post_incumbent_prob = self._sigmoid(post_incumbent_logit)
        post_candidate_prob = self._sigmoid(post_candidate_logit)
        return {
            "target_prob": candidate_example["target_prob"],
            "target_margin": target_margin,
            "reward": candidate_example["reward"],
            "delta_reward": candidate_example["delta_reward"],
            "pre_prob": candidate_prob,
            "post_prob": post_candidate_prob,
            "pre_logit": candidate_logit,
            "post_logit": post_candidate_logit,
            "keep_prob": incumbent_prob,
            "post_keep_prob": post_incumbent_prob,
            "pre_margin": pre_margin,
            "post_margin": post_candidate_prob - post_incumbent_prob,
            "pre_logit_margin": candidate_logit - incumbent_logit,
            "post_logit_margin": post_candidate_logit - post_incumbent_logit,
            "feature_norm": candidate_example["feature_norm"],
            "step_l2": float(np.linalg.norm(weight_step)),
            "selected": candidate_example["selected"],
            "role": candidate_example["role"],
            "candidate_pred": candidate_example["candidate_pred"],
        }

    def _update_score_pairwise(self, examples):
        if len(examples) < 2:
            return {
                "num_pairs": 0,
                "num_violations": 0,
                "step_l2": 0.0,
            }

        sorted_examples = sorted(
            examples,
            key=lambda item: float(item["reward"]),
            reverse=True,
        )
        total_step_l2 = 0.0
        num_pairs = 0
        num_violations = 0

        for better_idx in range(len(sorted_examples)):
            better = sorted_examples[better_idx]
            for worse_idx in range(better_idx + 1, len(sorted_examples)):
                worse = sorted_examples[worse_idx]
                if better["reward"] <= worse["reward"] + VERIFIER_ACCEPT_ZERO_DELTA_EPS:
                    continue

                better_logit = float(np.dot(self.score_weights, better["features"]) + self.score_bias)
                worse_logit = float(np.dot(self.score_weights, worse["features"]) + self.score_bias)
                pair_logit = better_logit - worse_logit
                pair_prob = self._sigmoid(pair_logit)
                error = pair_prob - 1.0
                feature_delta = better["features"] - worse["features"]
                weight_step = self.lr * error * feature_delta
                self.score_weights -= weight_step

                num_pairs += 1
                total_step_l2 += float(np.linalg.norm(weight_step))
                if pair_logit <= 0.0:
                    num_violations += 1

        return {
            "num_pairs": num_pairs,
            "num_violations": num_violations,
            "step_l2": total_step_l2,
        }

    def update_accept_batch(
        self,
        question,
        history,
        incumbent_text,
        incumbent_reward,
        candidate_updates,
    ):
        if incumbent_text is None or not candidate_updates:
            return None

        incumbent_example = self._prepare_score_example(
            question,
            incumbent_text,
            incumbent_reward,
            role="incumbent",
        )
        candidate_examples = []
        for item in candidate_updates:
            prepared = self._prepare_score_example(
                question,
                item.get("current_text"),
                item.get("reward"),
                candidate_pred=item.get("candidate_pred"),
                delta_reward=item.get("delta_reward"),
                selected=item.get("selected", False),
                role="candidate",
            )
            if prepared is not None:
                candidate_examples.append(prepared)

        if incumbent_example is None or not candidate_examples:
            return None

        margin_stats = [
            self._update_score_margin(incumbent_example, example)
            for example in candidate_examples
        ]
        pairwise_stats = self._update_score_pairwise(
            [incumbent_example, *candidate_examples]
        )

        post_keep_forward = self._forward_score(question, incumbent_text)

        selected_stats = next(
            (stat for stat in margin_stats if stat.get("selected")),
            None,
        )
        num_positive = sum(
            1
            for example in candidate_examples
            if example["delta_reward"] is not None
            and example["delta_reward"] > VERIFIER_ACCEPT_ZERO_DELTA_EPS
        )
        num_negative = sum(
            1
            for example in candidate_examples
            if example["delta_reward"] is not None
            and example["delta_reward"] < -VERIFIER_ACCEPT_ZERO_DELTA_EPS
        )
        num_zero = len(candidate_examples) - num_positive - num_negative
        num_supervised = len(margin_stats)
        margin_step_l2 = float(sum(stat["step_l2"] for stat in margin_stats))

        if selected_stats is not None:
            selected_text = next(
                (
                    example["answer_text"]
                    for example in candidate_examples
                    if example["selected"]
                ),
                None,
            )
            post_selected_forward = self._forward_score(question, selected_text)
            selected_stats = {
                **selected_stats,
                "post_prob": post_selected_forward["prob"],
                "post_logit": post_selected_forward["logit"],
                "post_keep_prob": post_keep_forward["prob"],
                "accept_threshold": 0.0,
                "accept_barrier": selected_stats["keep_prob"],
                "post_accept_barrier": post_keep_forward["prob"],
                "post_margin": post_selected_forward["prob"] - post_keep_forward["prob"],
                "post_logit_margin": post_selected_forward["logit"] - post_keep_forward["logit"],
            }

        return {
            "num_candidates": len(candidate_updates),
            "num_parseable": len(candidate_examples),
            "num_supervised": num_supervised,
            "num_skipped": 0,
            "num_positive": num_positive,
            "num_negative": num_negative,
            "num_zero": num_zero,
            "margin_step_l2": margin_step_l2,
            "pairwise_num_pairs": pairwise_stats["num_pairs"],
            "pairwise_num_violations": pairwise_stats["num_violations"],
            "pairwise_step_l2": pairwise_stats["step_l2"],
            "selected": selected_stats,
        }

    def update_accept(
        self,
        question,
        history,
        incumbent_text,
        current_text,
        candidate_pred,
        delta_reward,
    ):
        return None
