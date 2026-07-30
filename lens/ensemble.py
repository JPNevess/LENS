"""The LENS ensemble: dynamic selection, self-training and drift adaptation
driven by a single multi-target competence map.
"""
import collections

import numpy as np

from ._java import ensure_java_home
from ._vendor import ensure_vendor_path

ensure_java_home()
ensure_vendor_path()

from capymoa.classifier import HoeffdingAdaptiveTree
from capymoa.drift.detectors import ADWIN
from capymoa.instance import LabeledInstance
from river import tree as _river_tree

from MLHAT import MLHAT
from multioutput import BinaryRelevance as _BinaryRelevance

from .config import (
    CONFIG_12, PAPER_CONFIGS,
    INF_MARGIN, INF_MLHAT_A, INF_MMR, INF_NONE,
    TR_DISAGREE_C, TR_NONE, TR_SELF_A, TR_SELF_M, TR_TRAIN_C, TR_TRAIN_W,
    DIVERSITY_CORRELATION, DIVERSITY_DISAGREEMENT, DIVERSITY_DOUBLE_FAULT,
    DIVERSITY_KAPPA, DIVERSITY_Q_STATISTIC,
)


class LENS:
    """Semi-supervised streaming ensemble built on a multi-target competence map.

    Base learners are Hoeffding Adaptive Trees trained with random patches, that
    is feature subsampling combined with Poisson bagging. A multi-label
    meta-learner, the referee, predicts per instance whether each member is
    currently correct. Those estimates drive three mechanisms:

    * inference: members are scored by relevance and selected with maximal
      marginal relevance, so the selected subset is competent and complementary;
    * training: pseudo-labels for unlabelled instances are admitted and weighted
      by meta-learned reliability;
    * adaptation: ADWIN monitors a meta-learned signal and, on drift, lambda
      shifts towards diversity while weak members are replaced from a background
      pool.

    ``inference_mode`` and ``training_mode`` select which mechanisms are active;
    see ``lens.config``.
    """

    def __init__(self,
                 schema,
                 config=CONFIG_12,
                 ensemble_size=30,
                 window_size=1000,
                 lambda_param=0.5,
                 grace_period=50,
                 tie_threshold=0.05,
                 confidence=0.01,
                 leaf_prediction="NaiveBayesAdaptive",
                 pool_size=70,
                 pool_fresh_frac=0.5,
                 diversity_measure=DIVERSITY_DISAGREEMENT,
                 subspace_frac=0.6,
                 pseudo_conf_threshold=0.9,
                 pseudo_warmup_labels=200,
                 div_batch_n=200,
                 adwin_delta=0.002,
                 mlhat_alpha=0.05,
                 adapt_lambda=True,
                 lambda_trend_k=5.0,
                 lambda_stable=0.85,
                 lambda_relax=0.02,
                 k_floor_frac=0.5,
                 vote_temperature=0.5,
                 unsupervised_drift=False,
                 self_train_referee=False,
                 mmr_soft_weights=True,
                 inference_mode=None,
                 training_mode=None,
                 self_train_margin=0.5,
                 warmup_labeled=0,
                 referee_mode="mlhat",
                 seed=42):

        self.config = config
        if config in PAPER_CONFIGS:
            inf_default, tr_default = PAPER_CONFIGS[config]
        else:
            inf_default, tr_default = INF_MMR, TR_TRAIN_W
        self.inference_mode = inference_mode if inference_mode else inf_default
        self.training_mode  = training_mode  if training_mode  else tr_default

        self.use_mmr              = (self.inference_mode == INF_MMR)
        self.use_pseudo_label     = (self.training_mode != TR_NONE)
        self.use_fading_relevance = False
        self.self_train_referee   = bool(self_train_referee)
        self.self_train_margin    = float(self_train_margin)
        self.diversity_from_preds = False
        self.diversity_measure    = diversity_measure

        self.ensemble_size = ensemble_size
        self.window_size   = window_size
        self.seed          = seed
        self.schema        = schema
        self.rng           = np.random.default_rng(seed)

        self.lambda_param    = lambda_param
        self._lambda_initial = lambda_param
        self.adapt_lambda    = adapt_lambda
        self.adwin           = ADWIN(delta=adwin_delta)
        self.lambda_cooldown = 0
        self.acc_window      = collections.deque(maxlen=200)
        self._lambda_trend_k = lambda_trend_k
        self._acc_trend      = 0.0
        self._lambda_stable  = float(np.clip(lambda_stable, 0.05, 0.95))
        self._lambda_relax   = float(np.clip(lambda_relax, 0.0, 1.0))
        self._last_n_swapped = 0

        if hasattr(self.adwin, 'add_element'):
            self._adwin_update = self.adwin.add_element
        elif hasattr(self.adwin, 'update'):
            self._adwin_update = self.adwin.update
        elif hasattr(self.adwin, 'add_input'):
            self._adwin_update = self.adwin.add_input
        else:
            raise AttributeError(
                f"ADWIN has no update method. Available: "
                f"{[m for m in dir(self.adwin) if not m.startswith('_')]}")

        if hasattr(self.adwin, 'detected_change') and callable(self.adwin.detected_change):
            self._adwin_drift = self.adwin.detected_change
        elif hasattr(self.adwin, 'detected_change'):
            self._adwin_drift = lambda: self.adwin.detected_change
        elif hasattr(self.adwin, 'change_detected'):
            self._adwin_drift = lambda: self.adwin.change_detected
        else:
            raise AttributeError(
                f"ADWIN has no drift detection method. Available: "
                f"{[m for m in dir(self.adwin) if not m.startswith('_')]}")

        def _make_tree():
            return HoeffdingAdaptiveTree(
                schema=schema,
                grace_period=grace_period,
                tie_threshold=tie_threshold,
                confidence=confidence,
                leaf_prediction=leaf_prediction,
            )

        self._make_tree  = _make_tree
        self.members     = [_make_tree() for _ in range(ensemble_size)]

        self.pool_size     = pool_size
        self.pool_fresh_frac = float(np.clip(pool_fresh_frac, 0.0, 1.0))
        self.pool          = [_make_tree() for _ in range(pool_size)]
        self.pool_correct  = np.zeros((pool_size, window_size), dtype=int)
        self._pool_w_idx   = 0
        self._pool_w_cnt   = 0
        self.pool_accuracy = np.zeros(pool_size)

        self.n_features     = schema.get_num_attributes()
        self.subspace_size  = max(1, int(round(subspace_frac * self.n_features)))
        self._member_feat   = [self._sample_subspace() for _ in range(ensemble_size)]
        self._pool_feat     = [self._sample_subspace() for _ in range(pool_size)]

        self.pseudo_conf_threshold = pseudo_conf_threshold
        self.pseudo_warmup_labels  = pseudo_warmup_labels
        self._n_labeled            = 0
        self._n_total              = 0
        self.warmup_labeled        = int(warmup_labeled)

        self.use_unsup_drift = bool(unsupervised_drift)
        self.student         = _make_tree() if self.use_unsup_drift else None
        self._student_ready  = False
        self._last_drift     = False
        self._pending_drift  = False

        self.referee_mode = referee_mode
        if referee_mode == "binary_relevance":
            _base_hat = _river_tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=200, seed=seed)
            self.referee = _BinaryRelevance(_base_hat)
        elif referee_mode == "mlhat":
            self.referee = MLHAT()
        else:
            raise ValueError(f"unknown referee_mode: {referee_mode!r} "
                             "(use 'mlhat' or 'binary_relevance')")

        self.mlhat_acc_sliding = np.full(ensemble_size, 0.5)
        self._mlhat_alpha      = mlhat_alpha

        self._lbd_member_updates = 0
        self._lbd_n_instances    = 0
        self._lbd_gap_sum        = 0.0
        self._lbd_conf_sum       = 0.0
        self._lbd_pool_updates   = 0
        self._lbd_gap_pre_sum    = 0.0
        self._lbd_conf_pre_sum   = 0.0
        self._lbd_pre_count      = 0

        self.W_est      = np.zeros((ensemble_size, window_size), dtype=int)
        self.W_pred     = np.zeros((ensemble_size, window_size), dtype=int)
        self.w_est_idx  = 0
        self.w_est_cnt  = 0

        self.member_correct    = np.zeros((ensemble_size, window_size), dtype=int)
        self.member_win_idx    = np.zeros(ensemble_size, dtype=int)
        self.member_win_cnt    = np.zeros(ensemble_size, dtype=int)
        self.member_accuracy   = np.zeros(ensemble_size)

        self.r_correct  = np.zeros((ensemble_size, window_size), dtype=int)
        self.r_win_idx  = 0
        self.r_win_cnt  = 0
        self.r_accuracy = np.zeros(ensemble_size)
        self.current_k  = ensemble_size
        self.k_floor    = max(1, int(round(np.clip(k_floor_frac, 0.0, 1.0) * ensemble_size)))

        self._vote_temperature = float(np.clip(vote_temperature, 0.0, 1.0))

        self.mmr_soft_weights = bool(mmr_soft_weights) and self.use_mmr

        self._last_member_preds   = np.zeros(ensemble_size, dtype=int)
        self._last_member_margins = np.zeros(ensemble_size)
        self._last_member_conf    = np.zeros(ensemble_size)
        self._last_mmr_indices    = np.arange(ensemble_size)
        self._last_competence     = np.zeros(ensemble_size)
        self._last_pred_acc       = np.zeros(ensemble_size)

        self.warmup        = True
        self.warmup_count  = 0
        self.warmup_period = 500
        self.drift_count   = 0

        self._div_batch_n           = div_batch_n
        self._div_batch_counter     = 0
        self._div_alpha             = 0.2
        self._diversity_smooth      = np.zeros((ensemble_size, ensemble_size))
        self._diversity_initialized = False

    def _sample_subspace(self):
        """Binary mask over the features with ``subspace_size`` active entries."""
        mask = np.zeros(self.n_features, dtype=float)
        idx  = self.rng.choice(self.n_features, size=self.subspace_size, replace=False)
        mask[idx] = 1.0
        return mask

    def _masked_instance(self, x, mask, y):
        """Copy of the instance with the features outside the subspace zeroed."""
        return LabeledInstance.from_array(self.schema, x * mask, int(y))

    def _student_predict(self, instance):
        """Student prediction, used by the student-teacher drift signal."""
        if not self._student_ready:
            return -1
        try:
            votes = np.array(list(
                self.student.moa_learner.getVotesForInstance(instance.java_instance)
            ), dtype=float)
            return int(np.argmax(votes)) if votes.sum() > 0 else 0
        except Exception:
            return 0

    def _train_student(self, instance, teacher_label):
        """Train the student to imitate the ensemble prediction."""
        inst = LabeledInstance.from_array(self.schema, instance.x, int(teacher_label))
        self.student.train(inst)
        self._student_ready = True

    def _compute_diversity_matrix(self, W_matrix, w_cnt):
        """Pairwise diversity distance between learners over the correctness window."""
        n = self.ensemble_size

        if w_cnt == 0:
            return np.zeros((n, n))

        W_active = W_matrix[:, :w_cnt]
        N11 = W_active @ W_active.T
        num_ones  = np.sum(W_active, axis=1, keepdims=True)
        N10  = num_ones   - N11
        N01  = num_ones.T - N11
        N00  = w_cnt - N11 - N10 - N01
        N    = float(w_cnt)

        if self.diversity_measure == DIVERSITY_Q_STATISTIC:
            Q_num = (N11 * N00) - (N10 * N01)
            Q_den = (N11 * N00) + (N10 * N01)
            Q = np.ones((n, n))
            np.divide(Q_num, Q_den, out=Q, where=Q_den != 0)
            return Q

        elif self.diversity_measure == DIVERSITY_DISAGREEMENT:
            total = N11 + N10 + N01 + N00
            dis = np.zeros((n, n))
            np.divide(N10 + N01, total, out=dis, where=total != 0)
            return 1.0 - dis

        elif self.diversity_measure == DIVERSITY_KAPPA:
            P_obs = (N11 + N00) / N
            P_exp = ((N11 + N10) * (N11 + N01) + (N00 + N10) * (N00 + N01)) / (N * N)
            kappa = np.zeros((n, n))
            denom = 1.0 - P_exp
            np.divide(P_obs - P_exp, denom, out=kappa, where=denom != 0)
            return kappa

        elif self.diversity_measure == DIVERSITY_DOUBLE_FAULT:
            total = N11 + N10 + N01 + N00
            df = np.zeros((n, n))
            np.divide(N00, total, out=df, where=total != 0)
            return df

        elif self.diversity_measure == DIVERSITY_CORRELATION:
            denom = np.sqrt(
                (N11 + N10) * (N11 + N01) * (N10 + N00) * (N01 + N00)
            )
            rho = np.zeros((n, n))
            np.divide((N11 * N00) - (N10 * N01), denom, out=rho, where=denom != 0)
            return rho

        else:
            raise ValueError(f"Diversity measure desconhecida: {self.diversity_measure}")

    def _pred_agreement_matrix(self, W_pred, w_cnt):
        """Pairwise redundancy taken from the raw member predictions.

        ``agreement(i, j)`` is the fraction of the window where both members
        predicted the same class, so a high value means redundant, which is what MMR
        penalises. Computed as a sum of one-hot outer products.
        """
        n = self.ensemble_size
        if w_cnt == 0:
            return np.zeros((n, n))
        P = W_pred[:, :w_cnt]
        K = self.schema.get_num_classes()
        agree = np.zeros((n, n))
        for c in range(K):
            Ic = (P == c).astype(float)
            agree += Ic @ Ic.T
        return agree / float(w_cnt)

    def _refresh_diversity(self):
        """Recompute the diversity matrix and update its moving average."""
        if self.diversity_from_preds:
            D = self._pred_agreement_matrix(self.W_pred, self.w_est_cnt)
        else:
            D = self._compute_diversity_matrix(self.W_est, self.w_est_cnt)
        if not self._diversity_initialized:
            self._diversity_smooth      = D
            self._diversity_initialized = True
        else:
            self._diversity_smooth = (
                self._div_alpha * D
                + (1.0 - self._div_alpha) * self._diversity_smooth
            )

    def _apply_mmr_selection(self, competences):
        """Rank members by iterative maximal marginal relevance.

        Uses the cached diversity matrix, refreshed every ``div_batch_n`` instances.
        With MMR off the ranking is by relevance alone; before the first batch the
        diversity term is still zero, which amounts to the same thing.
        """
        if not self.use_mmr:
            return np.argsort(competences)[::-1].astype(int)

        sim_matrix = self._diversity_smooth

        selected = []
        first_idx = int(np.argmax(competences))
        selected.append(first_idx)

        max_redundancy  = np.full(self.ensemble_size, -1.0)
        unselected_mask = np.ones(self.ensemble_size, dtype=bool)
        unselected_mask[first_idx] = False

        for _ in range(self.ensemble_size - 1):
            last_selected  = selected[-1]
            current_sim    = sim_matrix[:, last_selected]
            max_redundancy = np.maximum(max_redundancy, current_sim)

            mmr_scores = (
                self.lambda_param * competences
                - (1.0 - self.lambda_param) * max_redundancy
            )
            mmr_scores[~unselected_mask] = -np.inf

            best_candidate = int(np.argmax(mmr_scores))
            selected.append(best_candidate)
            unselected_mask[best_candidate] = False

        return np.array(selected, dtype=int)

    def predict(self, instance, x_dict):
        """Predict with the ensemble. Returns (prediction, per-member competence)."""
        margins   = np.zeros(self.ensemble_size)
        confs     = np.zeros(self.ensemble_size)
        all_preds = np.zeros(self.ensemble_size, dtype=int)

        x = instance.x
        for i, member in enumerate(self.members):
            try:
                minst       = self._masked_instance(x, self._member_feat[i], 0)
                votes_array = member.moa_learner.getVotesForInstance(minst.java_instance)
                votes       = np.array(list(votes_array), dtype=float)
                total_v     = np.sum(votes)
                if total_v > 0:
                    probs  = votes / total_v
                    pred   = int(np.argmax(probs))
                    conf   = float(probs[pred])
                    margin = (np.sort(probs)[::-1][0] - np.sort(probs)[::-1][1]) if len(probs) > 1 else 1.0
                else:
                    pred, margin, conf = 0, 0.0, 0.0
            except Exception:
                pred, margin, conf = 0, 0.0, 0.0

            all_preds[i] = pred
            margins[i]   = margin
            confs[i]     = conf

        self._last_member_preds   = all_preds
        self._last_member_margins = margins
        self._last_member_conf    = confs

        referee_proba = self.referee.predict_proba_one(x_dict)
        pred_acc      = np.zeros(self.ensemble_size)
        est_correct   = np.zeros(self.ensemble_size, dtype=int)

        for i in range(self.ensemble_size):
            label_name = f"model_{i}"
            if label_name in referee_proba:
                p_err = referee_proba[label_name].get(1, 0.0)
            else:
                p_err = 0.5
            pred_acc[i]    = 1.0 - p_err
            est_correct[i] = 1 if p_err < 0.5 else 0

        self._last_pred_acc = pred_acc

        self.mlhat_acc_sliding = (
            (1.0 - self._mlhat_alpha) * self.mlhat_acc_sliding
            + self._mlhat_alpha * pred_acc
        )

        epsilon = self.rng.uniform(0, 1e-6, size=self.ensemble_size)

        if self.warmup:
            effective_acc = np.ones(self.ensemble_size)
        elif self.use_fading_relevance:
            effective_acc = self.mlhat_acc_sliding
        else:
            effective_acc = pred_acc

        m = self.inference_mode
        if m == INF_NONE:
            competences = np.ones(self.ensemble_size) + epsilon
        elif m == INF_MARGIN:
            competences = margins + epsilon
        elif m == INF_MLHAT_A:
            competences = effective_acc + epsilon
        else:
            competences = np.sqrt(margins * effective_acc) + epsilon
        self._last_competence = competences

        self._last_mmr_indices = self._apply_mmr_selection(competences)

        final_pred = self._final_prediction()

        for i in range(self.ensemble_size):
            self.W_est[i, self.w_est_idx]  = est_correct[i]
            self.W_pred[i, self.w_est_idx] = all_preds[i]

        if self.use_unsup_drift:
            st_pred = self._student_predict(instance)
            if st_pred >= 0:
                drift = self._update_lambda(int(st_pred == final_pred))
                if drift:
                    self._pending_drift = True
            self._train_student(instance, final_pred)

        return final_pred

    def _topk_vote_weights(self):
        """Return the selected members and their vote weights.

        Weights are relevance raised to ``vote_temperature``. With soft weighting on,
        members are visited in MMR order and each one transfers
        ``sim(i, j) * w_i * w_j`` to the better-ranked members it is redundant with.
        Diversity therefore affects every prediction, not only which members enter
        the top-K.
        """
        if self.inference_mode == INF_NONE:
            allm = np.arange(self.ensemble_size)
            return allm, np.ones(self.ensemble_size)

        top_k   = self._last_mmr_indices[:self.current_k]
        weights = np.power(
            np.clip(self._last_competence[top_k], 1e-12, None),
            self._vote_temperature,
        )
        if self.mmr_soft_weights and self._diversity_initialized and len(top_k) > 1:
            w   = weights / max(float(weights.sum()), 1e-12)
            sim = self._diversity_smooth
            for i in range(1, len(top_k)):
                eta    = sim[top_k[i], top_k[:i]] * w[:i] * w[i]
                w[:i] += eta
                w[i]   = max(w[i] - float(eta.sum()), 0.0)
            weights = w
        return top_k, weights

    def _final_prediction(self):
        """Final prediction: relevance-weighted vote over the selected members."""
        top_k, weights = self._topk_vote_weights()
        return _weighted_vote(self._last_member_preds[top_k], weights)

    def _final_prediction_conf(self):
        """As ``_final_prediction``, also returning the winner's share of the weight."""
        top_k, weights = self._topk_vote_weights()
        return _weighted_vote_conf(self._last_member_preds[top_k], weights)

    def train(self, instance, x_dict, y_true):
        """Training step for an instance whose true label is available."""

        self._n_labeled += 1
        self._n_total   += 1

        self._update_dynamic_k(y_true, use_label=True)

        if self.use_unsup_drift:
            drift_detected = self._pending_drift
            self._pending_drift = False
        else:
            ens_pred = self._final_prediction()
            drift_detected = self._update_lambda(int(ens_pred == y_true))

        poisson_lambda = 6.0

        actual_errors = {}
        for i, member in enumerate(self.members):
            is_correct = int(self._last_member_preds[i] == y_true)

            self.W_est[i, self.w_est_idx] = is_correct

            w_idx = self.member_win_idx[i]
            self.member_correct[i, w_idx]  = is_correct
            self.member_win_idx[i]         = (w_idx + 1) % self.window_size
            self.member_win_cnt[i]         = min(self.member_win_cnt[i] + 1, self.window_size)
            self.member_accuracy[i]        = (
                np.sum(self.member_correct[i, :self.member_win_cnt[i]])
                / self.member_win_cnt[i]
            )

            actual_errors[f"model_{i}"] = 1 - is_correct

            minst  = self._masked_instance(instance.x, self._member_feat[i], y_true)
            weight = int(self.rng.poisson(poisson_lambda))
            if weight > 0:
                _train_weighted(member, minst, weight)

        self._train_pool(instance, y_true, poisson_lambda, drift_detected,
                         is_labeled=True)

        self.w_est_idx  = (self.w_est_idx + 1) % self.window_size
        self.w_est_cnt  = min(self.w_est_cnt + 1, self.window_size)

        self._div_batch_counter += 1
        if self._div_batch_counter >= self._div_batch_n:
            self._div_batch_counter = 0
            self._refresh_diversity()

        if actual_errors:
            self.referee.learn_one(x_dict, actual_errors)

        self.warmup_count += 1
        if self.warmup and self.warmup_count >= self.warmup_period:
            self.warmup = False

    def train_unsupervised(self, instance, x_dict):
        """Training step for an unlabelled instance.

        Trains on the pseudo-label and never on the true label: a fresh
        ``LabeledInstance`` is built so the real label cannot leak into the update.
        """

        self._n_total += 1

        self.w_est_idx = (self.w_est_idx + 1) % self.window_size
        self.w_est_cnt = min(self.w_est_cnt + 1, self.window_size)

        self._div_batch_counter += 1
        if self._div_batch_counter >= self._div_batch_n:
            self._div_batch_counter = 0
            self._refresh_diversity()

        self.warmup_count += 1
        if self.warmup and self.warmup_count >= self.warmup_period:
            self.warmup = False

        if not self.use_pseudo_label:
            return

        if self.training_mode in (TR_SELF_M, TR_TRAIN_C, TR_SELF_A):
            self._train_self_M(instance)
        else:
            self._train_learn_by_disagreement(instance)

        if self._n_labeled < self.pseudo_warmup_labels:
            return
        if self.use_unsup_drift:
            drift_detected = self._pending_drift
            self._pending_drift = False
        else:
            drift_detected = False
        pseudo_label, conf = self._final_prediction_conf()

        if self.self_train_referee and conf >= self.pseudo_conf_threshold:
            # Label density excludes the fully labelled warm-up window, so
            # pseudo-labels do not start at almost full weight.
            density = max(0, self._n_labeled - self.warmup_labeled) / max(1, self._n_total)
            if density > 0:
                pseudo_err = {
                    f"model_{i}": int(self._last_member_preds[i] != pseudo_label)
                    for i in range(self.ensemble_size)
                }
                self.referee.learn_one(x_dict, pseudo_err, sample_weight=float(density))

        if conf < self.pseudo_conf_threshold:
            if drift_detected:
                self._train_pool(instance, pseudo_label, 0.0,
                                 drift_detected, is_labeled=False)
            return
        self._train_pool(instance, pseudo_label, 2.0,
                         drift_detected, is_labeled=False)

    def _admission_signals(self):
        """Per-member admission score and pseudo-label for the current instance.

        Returns ``(scores, pseudo_labels, signal_name)``, or None when the training
        mode admits nothing. The score decides whether a pseudo-label is accepted and
        the pseudo-label is the one that would train that member, so comparing both
        against the true label offline measures whether the signal predicts
        pseudo-label correctness. NaN marks members without a usable consensus.

            self_train_M   score = M,                 pseudo = own prediction
            self_train_A   score = A_hat,              pseudo = own prediction
            train_C        score = sqrt(A_hat * M),    pseudo = own prediction
            disagree_c     score = c,                  pseudo = majority of the others
            train_w        score = sqrt(A_bar * c),    pseudo = confident majority
        """
        preds = np.asarray(self._last_member_preds, dtype=int)
        n     = self.ensemble_size
        tm    = self.training_mode
        if tm == TR_SELF_M:
            return self._last_member_margins.astype(float), preds, "M"
        if tm == TR_SELF_A:
            return self._last_pred_acc.astype(float), preds, "A"
        if tm == TR_TRAIN_C:
            C = np.sqrt(np.clip(self._last_pred_acc * self._last_member_margins,
                                0.0, None))
            return C.astype(float), preds, "C"
        if tm in (TR_DISAGREE_C, TR_TRAIN_W):
            hybrid = (tm == TR_TRAIN_W)
            M      = self._last_member_margins
            A_bar  = self.mlhat_acc_sliding
            voter  = (M > self.self_train_margin) if hybrid else np.ones(n, dtype=bool)
            n_classes = int(preds.max()) + 1 if n else 1
            counts_all = np.bincount(preds[voter], minlength=n_classes).astype(float)
            scores  = np.empty(n)
            plabels = np.empty(n, dtype=int)
            for l in range(n):
                c = counts_all.copy()
                if voter[l]:
                    c[preds[l]] -= 1.0
                tot = c.sum()
                if tot <= 0:
                    scores[l] = np.nan; plabels[l] = preds[l]; continue
                lab = int(c.argmax())
                c_L = c[lab] / tot
                plabels[l] = lab
                scores[l]  = np.sqrt(max(A_bar[l], 0.0) * c_L) if hybrid else c_L
            return scores, plabels, ("w" if hybrid else "c")
        return None

    def _train_self_M(self, instance):
        """Self-training: a member trains on its own prediction when confident.

        The gate and the weight are what separate the modes:

            self_train_M   gate M >= tau,      weight M * label density
            train_C        gate M >= tau,      weight sqrt(A_hat * M) * label density
            self_train_A   gate A_hat >= tau,  weight A_hat * label density
        """
        if self._n_labeled < self.pseudo_warmup_labels:
            return
        margins = self._last_member_margins
        preds   = self._last_member_preds
        A       = self._last_pred_acc
        density = max(0, self._n_labeled - self.warmup_labeled) / max(1, self._n_total)
        if density <= 0:
            return
        use_C = (self.training_mode == TR_TRAIN_C)
        use_A = (self.training_mode == TR_SELF_A)
        if use_C:
            C = np.sqrt(np.clip(A * margins, 0.0, None))
        gate = A if use_A else margins
        n_upd = 0
        for l in range(self.ensemble_size):
            if gate[l] < self.self_train_margin:
                continue
            score = A[l] if use_A else (C[l] if use_C else margins[l])
            w = float(score * density)
            if w <= 0:
                continue
            inst = self._masked_instance(instance.x, self._member_feat[l], preds[l])
            _train_weighted(self.members[l], inst, w)
            n_upd += 1
        if n_upd:
            self._lbd_member_updates += n_upd
            self._lbd_n_instances    += 1

    def _train_learn_by_disagreement(self, instance):
        """Learn-by-disagreement pseudo-labelling.

            disagree_c   voters are all other members; pseudo-label by simple
                 majority, c is the agreeing fraction, weight is c
            train_w      voters are the other members with margin above tau;
                 pseudo-label by majority of those, c is the agreeing
                 fraction among them, weight is sqrt(A_bar * c)

        A member is updated only when it sits in the weaker half of the ensemble, the
        rest of the ensemble is more confident than the member itself, the two
        disagree, and c reaches the admission threshold. The weight is then scaled by
        the label density. Pre-threshold sums over all members are accumulated for
        diagnostics, where the confidence gap may be negative.
        """
        if self._n_labeled < self.pseudo_warmup_labels:
            return

        preds  = self._last_member_preds
        confs  = self._last_member_conf
        M      = self._last_member_margins
        A_bar  = self.mlhat_acc_sliding
        n      = self.ensemble_size
        if n <= 1:
            return

        density = max(0, self._n_labeled - self.warmup_labeled) / max(1, self._n_total)
        if density <= 0:
            return

        hybrid = (self.training_mode == TR_TRAIN_W)
        voter = (M > self.self_train_margin) if hybrid else np.ones(n, dtype=bool)
        n_classes = int(preds.max()) + 1
        counts_all = np.bincount(preds[voter], minlength=n_classes).astype(float)

        med_acc = float(np.median(self.member_accuracy))

        n_upd = 0
        for l in range(n):
            counts = counts_all.copy()
            if voter[l]:
                counts[preds[l]] -= 1.0
            total = counts.sum()
            if total <= 0:
                continue
            pseudo_label = int(np.argmax(counts))
            c_L_hat      = float(counts[pseudo_label] / total)
            c_l          = float(confs[l])

            self._lbd_gap_pre_sum  += c_L_hat - c_l
            self._lbd_conf_pre_sum += c_L_hat
            self._lbd_pre_count    += 1

            if self.member_accuracy[l] > med_acc:
                continue
            if c_L_hat <= c_l:
                continue
            if pseudo_label == preds[l]:
                continue
            if c_L_hat < self.pseudo_conf_threshold:
                continue

            if hybrid:
                w = float(np.sqrt(max(A_bar[l], 0.0) * c_L_hat) * density)
            else:
                w = float(c_L_hat * density)
            if w <= 0:
                continue

            pseudo_inst = self._masked_instance(instance.x, self._member_feat[l], pseudo_label)
            _train_weighted(self.members[l], pseudo_inst, w)

            n_upd += 1
            self._lbd_gap_sum  += c_L_hat - c_l
            self._lbd_conf_sum += c_L_hat

        if n_upd:
            self._lbd_member_updates += n_upd
            self._lbd_n_instances    += 1

    def _update_dynamic_k(self, y_true, use_label=True):
        """Pick the K that maximises windowed accuracy over the ranked members."""
        votes = collections.defaultdict(float)
        for k_idx in range(self.ensemble_size):
            member_idx = self._last_mmr_indices[k_idx]
            pred       = self._last_member_preds[member_idx]
            weight     = self._last_competence[member_idx] ** self._vote_temperature
            votes[pred] += weight
            k_pred     = int(max(votes.items(), key=lambda x: x[1])[0])
            self.r_correct[k_idx, self.r_win_idx] = 1 if k_pred == y_true else 0

        self.r_win_idx  = (self.r_win_idx + 1) % self.window_size
        self.r_win_cnt  = min(self.r_win_cnt + 1, self.window_size)
        self.r_accuracy = (
            np.sum(self.r_correct[:, :self.r_win_cnt], axis=1) / self.r_win_cnt
        )

        best_k_idx    = int(np.argmax(self.r_accuracy))
        candidate_k   = (
            self.ensemble_size
            if self.r_accuracy[best_k_idx] == 0.0
            else best_k_idx + 1
        )
        smoothed       = int(round(0.9 * self.current_k + 0.1 * candidate_k))
        self.current_k = int(np.clip(smoothed, self.k_floor, self.ensemble_size))

    def _update_lambda(self, is_correct_ensemble: int) -> bool:
        """Feed ADWIN and update lambda. Returns True when drift is detected."""
        self._adwin_update(is_correct_ensemble)
        self.acc_window.append(is_correct_ensemble)
        drift_detected = bool(self._adwin_drift())

        if drift_detected:
            self.drift_count += 1

        if self.adapt_lambda:
            if drift_detected:
                self.lambda_param = float(np.clip(0.5 + self._lambda_trend_k * self._acc_trend, 0.05, 0.95))
            else:
                self.lambda_param = float(np.clip(
                    self.lambda_param + self._lambda_relax * (self._lambda_stable - self.lambda_param),
                    0.05, 0.95))

        return drift_detected

    def _train_pool(self, instance, label, poisson_lambda, drift_detected,
                    is_labeled=True):
        """Train the background pool on every instance.

        Labelled instances train on the true label with a higher weight, unlabelled
        ones on the ensemble consensus, like the active members; otherwise at a 5%
        label rate the reserve would learn twenty times slower and be stale exactly
        when it is needed. Pool accuracy is measured on labelled instances only, so it
        stays comparable with member accuracy.

        On drift the weakest active members are replaced by pool candidates that are
        both accurate and diverse, and fresh trees are injected into the weakest pool
        slots.
        """
        x = instance.x
        for j, pool_member in enumerate(self.pool):
            pminst = self._masked_instance(x, self._pool_feat[j], label)

            if is_labeled:
                try:
                    votes_arr = pool_member.moa_learner.getVotesForInstance(pminst.java_instance)
                    votes     = np.array(list(votes_arr), dtype=float)
                    pool_pred = int(np.argmax(votes)) if votes.sum() > 0 else 0
                except Exception:
                    pool_pred = 0
                self.pool_correct[j, self._pool_w_idx] = int(pool_pred == label)

            pool_weight = int(self.rng.poisson(poisson_lambda))
            if pool_weight > 0:
                _train_weighted(pool_member, pminst, pool_weight)
                if not is_labeled:
                    self._lbd_pool_updates += 1

        if is_labeled:
            self._pool_w_idx = (self._pool_w_idx + 1) % self.window_size
            self._pool_w_cnt = min(self._pool_w_cnt + 1, self.window_size)
            if self._pool_w_cnt:
                self.pool_accuracy = self.pool_correct[:, :self._pool_w_cnt].sum(axis=1) / self._pool_w_cnt

        if drift_detected and self.pool_size > 0:
            cand         = self._select_pool_members(min(self.ensemble_size, self.pool_size))
            active_order = list(np.argsort(self.member_accuracy))
            n_swapped    = 0
            for ai, pj in zip(active_order, cand):
                if self.pool_accuracy[pj] <= self.member_accuracy[ai]:
                    continue
                self.members[ai], self.pool[pj] = self.pool[pj], self.members[ai]
                self._member_feat[ai], self._pool_feat[pj] = (
                    self._pool_feat[pj], self._member_feat[ai])
                self.member_correct[ai]    = 0
                self.member_win_idx[ai]    = 0
                self.member_win_cnt[ai]    = 0
                self.member_accuracy[ai]   = 0.0
                self.W_est[ai, :]          = 0
                self.W_pred[ai, :]         = 0
                self.mlhat_acc_sliding[ai] = 0.5
                self.pool_correct[pj, :]   = 0
                self.pool_accuracy[pj]     = 0.0
                n_swapped += 1
            self._last_n_swapped = n_swapped

            n_fresh = int(round(self.pool_fresh_frac * self.pool_size))
            if n_fresh > 0:
                worst = np.argsort(self.pool_accuracy)[:n_fresh]
                for pj in worst:
                    self.pool[pj]            = self._make_tree()
                    self._pool_feat[pj]      = self._sample_subspace()
                    self.pool_correct[pj, :] = 0
                    self.pool_accuracy[pj]   = 0.0

    def _pool_redundancy_matrix(self):
        """Pairwise redundancy in the pool, ``1 - disagreement``; high means interchangeable."""
        n, cnt = self.pool_size, self._pool_w_cnt
        if cnt == 0:
            return np.zeros((n, n))
        W   = self.pool_correct[:, :cnt].astype(float)
        N11 = W @ W.T
        ones = W.sum(axis=1, keepdims=True)
        N10 = ones - N11
        N01 = ones.T - N11
        dis = (N10 + N01) / cnt
        return 1.0 - dis

    def _select_pool_members(self, n_sel):
        """Greedily pick ``n_sel`` pool members by accuracy penalised by redundancy
        against the ones already picked.
        """
        n   = self.pool_size
        acc = self.pool_accuracy.copy()
        red = self._pool_redundancy_matrix()
        w   = float(self.lambda_param)
        first = int(np.argmax(acc))
        selected = [first]
        avail = np.ones(n, dtype=bool); avail[first] = False
        max_red = red[:, first].copy()
        while len(selected) < n_sel:
            score = w * acc - (1.0 - w) * max_red
            score[~avail] = -np.inf
            nxt = int(np.argmax(score))
            selected.append(nxt); avail[nxt] = False
            max_red = np.maximum(max_red, red[:, nxt])
        return selected

    @staticmethod
    def _mean_disagreement(M, cnt):
        """Mean fraction of learner pairs that disagree on correctness over the window.

        In [0, 1], where higher is more diverse. Averaged over off-diagonal pairs.
        """
        if cnt is None or cnt <= 0:
            return float("nan")
        W   = M[:, :cnt].astype(float)
        n   = W.shape[0]
        if n <= 1:
            return float("nan")
        N11 = W @ W.T
        ones = W.sum(axis=1, keepdims=True)
        N10 = ones   - N11
        N01 = ones.T - N11
        dis = (N10 + N01) / cnt
        off = dis.sum() - np.trace(dis)
        return float(off / max(1, n * n - n))

    def _snapshot_metrics(self):
        """Instantaneous diagnostics, sampled into the run history."""
        active_real_acc = float(np.mean(self.member_accuracy))
        active_pred_acc = float(np.mean(self.mlhat_acc_sliding))
        active_real_div = self._mean_disagreement(
            self.member_correct, int(self.member_win_cnt.min()))
        if self._diversity_initialized:
            sim = self._diversity_smooth
            m   = sim.shape[0]
            off = (sim.sum() - np.trace(sim)) / max(1, m * m - m)
            active_pred_div = float(np.clip(1.0 - off, 0.0, 1.0))
        else:
            active_pred_div = float("nan")
        pool_real_acc = float(np.mean(self.pool_accuracy))
        pool_real_div = self._mean_disagreement(self.pool_correct, self._pool_w_cnt)

        Ml = np.asarray(self._last_member_margins, dtype=float)
        Al = np.asarray(self._last_pred_acc, dtype=float)
        comp = np.sqrt(np.clip(Al * Ml, 0.0, None))
        n = self.ensemble_size
        if n > 1 and comp.sum() > 0:
            c_L_hat = (comp.sum() - comp) / (n - 1)
            w_mean  = float(np.mean(np.sqrt(np.clip(
                self.mlhat_acc_sliding * c_L_hat, 0.0, None))))
        else:
            w_mean  = float("nan")

        upd = self._lbd_member_updates
        lbd_gap_mean  = (self._lbd_gap_sum  / upd) if upd else float("nan")
        lbd_conf_mean = (self._lbd_conf_sum / upd) if upd else float("nan")
        pre = self._lbd_pre_count
        lbd_gap_pre_mean  = (self._lbd_gap_pre_sum  / pre) if pre else float("nan")
        lbd_conf_pre_mean = (self._lbd_conf_pre_sum / pre) if pre else float("nan")

        return {
            "active_real_acc": active_real_acc,
            "active_pred_acc": active_pred_acc,
            "active_real_div": active_real_div,
            "active_pred_div": active_pred_div,
            "pool_real_acc"  : pool_real_acc,
            "pool_real_div"  : pool_real_div,
            "K"              : int(self.current_k),
            "lambda"         : float(self.lambda_param),
            "drift_count"    : int(self.drift_count),
            "n_swapped"      : int(self._last_n_swapped),
            "mean_margin"    : float(np.mean(Ml)),
            "mean_A_inst"    : float(np.mean(Al)),
            "mean_C"         : float(np.mean(comp)),
            "mean_w"         : w_mean,
            "lbd_updates"    : int(self._lbd_member_updates),
            "lbd_instances"  : int(self._lbd_n_instances),
            "lbd_gap_mean"   : lbd_gap_mean,
            "lbd_conf_mean"  : lbd_conf_mean,
            "lbd_gap_pre_mean" : lbd_gap_pre_mean,
            "lbd_conf_pre_mean": lbd_conf_pre_mean,
            "lbd_pool_updates": int(self._lbd_pool_updates),
        }


def _train_weighted(member, instance, w: float):
    java_inst = instance.java_instance
    java_inst.setWeight(float(w))
    member.moa_learner.trainOnInstance(java_inst)
    # The Java instance is shared across members, so always restore the weight.
    java_inst.setWeight(1.0)


def _weighted_vote(preds, weights):
    """Class with the largest total weight, or 0 when there are no votes."""
    votes = collections.Counter()
    for pred, weight in zip(preds, weights):
        votes[int(pred)] += float(weight)
    return votes.most_common(1)[0][0] if votes else 0


def _weighted_vote_conf(preds, weights):
    """As ``_weighted_vote``, returning ``(class, share of the total weight)``."""
    votes = collections.Counter()
    total = 0.0
    for pred, weight in zip(preds, weights):
        w = float(weight)
        votes[int(pred)] += w
        total += w
    if not votes or total <= 0:
        return 0, 0.0
    cls, w_cls = votes.most_common(1)[0]
    return cls, w_cls / total
