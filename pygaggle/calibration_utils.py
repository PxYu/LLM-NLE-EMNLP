"""
Utility functions for calibration.
"""

import torch
import random
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, ndcg_score
from matplotlib import pyplot as plt
import torch.nn.functional as F


def format_multigrade(
    queries,
    qrels,
    scored_qd,
    scale_score_to_max=True,
    binarize_labels=False,
    max_grade=3,
    reverse_logit=False,  # by default, first logit is False
    downsample_negative=None,
):
    """
    build {q: {"docids": [], "scores": [], "labels": [], "logits": []}}
    """
    ret = {}
    for q in queries:
        candidate_q_s_l = scored_qd[q]
        qrels_ = qrels[q]
        ret[q] = {"docids": [], "scores": [], "labels": [], "logits": []}

        for d, s, l in candidate_q_s_l:
            if downsample_negative is not None:
                if d not in qrels_:
                    if random.random() >= downsample_negative:
                        continue

            ret[q]["docids"].append(d)

            if scale_score_to_max:
                ret[q]["scores"].append(s * max_grade)
            else:
                ret[q]["scores"].append(s)

            if reverse_logit:
                ret[q]["logits"].append(l[::-1])
            else:
                ret[q]["logits"].append(l)

            if d in qrels_:
                if binarize_labels:
                    if qrels_[d] > 0:
                        ret[q]["labels"].append(1)
                    else:
                        ret[q]["labels"].append(0)
                else:
                    ret[q]["labels"].append(qrels_[d])
            else:
                ret[q]["labels"].append(0)

        if len(ret[q]["docids"]) <= 1:
            del ret[q]

    return ret


def convert_samples_to_flat_format(samples, key="scores"):
    """
    convert samples to flat format
    """
    all_y, all_p = [], []
    for dct in samples.values():
        all_y += dct["labels"]
        all_p += dct[key]
    return all_y, all_p


class ECE_Calculator_Yan:
    # ECE calculation based on Equation 13 of "Scale Calibration of Deep Ranking Models"

    def __init__(self, samples, num_bins=10):
        self.samples = samples
        self.num_bins = num_bins

    def divide_into_bins(self, input_list, corresponding_list):
        combined_lists = list(zip(input_list, corresponding_list))
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
        return [
            x.tolist() for x in np.array_split(sorted_combined_lists, self.num_bins)
        ]

    def ece_score_per_query(self, predictions, labels):
        ece_per_q = 0
        dq = len(predictions)
        bins = self.divide_into_bins(predictions, labels)
        for bin_ in bins:
            bin_error = np.abs(
                np.mean([x[0] for x in bin_]) - np.mean([x[1] for x in bin_])
            )
            ece_per_q += bin_error * len(bin_) / dq
        return ece_per_q

    def calculate(self, key):
        return np.mean(
            [
                self.ece_score_per_query(v[key], v["labels"])
                for v in self.samples.values()
            ]
        )


class ECE_Calculator:
    def __init__(self, n_bins, min_pred, max_pred):
        bin_boundaries = torch.linspace(min_pred, max_pred, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calculate(self, confidences, labels):
        self.bin_truth = []
        self.bin_confidence = []
        self.bin_prob = []
        self.bin_ece = []

        confidences = torch.tensor(confidences)
        labels = torch.tensor(labels)

        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                self.bin_prob.append(prop_in_bin.item())
                avg_truth_in_bin = labels[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                self.bin_truth.append(avg_truth_in_bin.item())
                self.bin_confidence.append(avg_confidence_in_bin.item())
                bin_ece = (
                    torch.abs(avg_confidence_in_bin - avg_truth_in_bin) * prop_in_bin
                )
                ece += bin_ece
                self.bin_ece.append(bin_ece)

        return ece.item()


def eval_model_or_param_on_samples(
    model_type, model_or_param, samples, max_grade, skip_query_level_ece
):
    for q, d in samples.copy().items():
        if model_type == "ir":
            # same scale as training labels
            p_calibrated = model_or_param.transform(d["scores"])
            p_calibrated[np.isnan(p_calibrated)] = 0
        elif model_type == "lr":
            """
            in terms of multi-grade regression (framed as multi-class classification),
            we get the probability of each grade, and then take the weighted sum
            """
            p = model_or_param.predict_proba(np.asarray(d["scores"]).reshape(-1, 1))
            num_grade = p.shape[1]
            scales = np.asarray(list(range(num_grade))).reshape(-1, 1)
            p_calibrated = p.dot(scales).squeeze()
            p_calibrated[np.isnan(p_calibrated)] = 0
        elif model_type == "ts":
            p_calibrated = (
                F.softmax(
                    temperature_scale(torch.tensor(d["logits"]), model_or_param), dim=1
                )[:, 1]
                * max_grade
            )
            p_calibrated = p_calibrated.tolist()
        else:
            assert False

        samples[q]["calibrated_scores"] = list(p_calibrated)
        samples[q]["ndcg_before"] = ndcg_score(
            np.asarray([d["labels"]]), np.asarray([d["scores"]]), ignore_ties=True
        )
        samples[q]["ndcg_after"] = ndcg_score(
            np.asarray([d["labels"]]),
            np.asarray([list(p_calibrated)]),
            ignore_ties=True,
        )

    y, p = convert_samples_to_flat_format(samples, key="scores")
    _, calibrated_p = convert_samples_to_flat_format(samples, key="calibrated_scores")

    print("\tBEFORE \t AFTER")

    ndcg_before = np.mean([v["ndcg_before"] for v in samples.values()])
    ndcg_after = np.mean([v["ndcg_after"] for v in samples.values()])
    print(f"nDCG:\t{ndcg_before:.3f}\t{ndcg_after:.3f}")

    if max_grade > 1:
        mse_before = mean_squared_error(y, p)
        mse_after = mean_squared_error(y, calibrated_p)
        print(f"MSE:\t{mse_before:.3f}\t{mse_after:.3f}")
    else:
        logloss_before = log_loss(y, p)
        logloss_after = log_loss(y, calibrated_p)
        print(f"LogLoss:{logloss_before:.3f}\t{logloss_after:.3f}")

    if not skip_query_level_ece:
        ece_cal_yan = ECE_Calculator_Yan(samples)
        ece_before = ece_cal_yan.calculate("scores")
        ece_after = ece_cal_yan.calculate("calibrated_scores")
        print(f"ECE_y:\t{ece_before:.3f}\t{ece_after:.3f}")

    ece_cal = ECE_Calculator(15, max_grade)
    macro_ece_before = ece_cal.calculate(p, y)
    x1, y1, p1 = ece_cal.bin_confidence, ece_cal.bin_truth, ece_cal.bin_prob
    macro_ece_after = ece_cal.calculate(calibrated_p, y)
    x2, y2, p2 = ece_cal.bin_confidence, ece_cal.bin_truth, ece_cal.bin_prob
    print(f"ECE:\t{macro_ece_before:.3f}\t{macro_ece_after:.3f}")

    plt.plot(
        range(0, max_grade + 1),
        range(0, max_grade + 1),
        label="ideal",
        linestyle="--",
        color="gray",
    )
    plt.plot(x1, y1, label="before", marker="o", color="b")
    plt.plot(x1, p1, color="b", alpha=0.2)
    plt.plot(x2, y2, label="after", marker="^", color="r")
    plt.plot(x2, p2, color="r", alpha=0.2)
    plt.xlabel("avg. conf in bin")
    plt.ylabel("avg. truth in bin")
    plt.legend()
    plt.show()


def temperature_scale(logits, temperature):
    return logits / temperature


class Temperature_Scaling:
    def __init__(
        self,
        training_logits=None,
        training_y=None,
        max_grade=1,
        learning_rate=1e-2,
        num_epochs=10000,
        init_t=1.0,
        is_regression=True,
    ):
        self.training_logits = training_logits
        self.training_y = training_y
        self.max_grade = max_grade
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.init_t = init_t
        self.is_regression = is_regression

    def nll_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)
        # return F.nll_loss(outputs, labels)

    def mse_loss(self, outputs, labels):
        return F.mse_loss(outputs, labels)

    def train_temperature_scaling(self):
        labels = torch.tensor(self.training_y, dtype=torch.float32)
        logits = torch.tensor(self.training_logits, requires_grad=True)
        temperature = torch.tensor(self.init_t, requires_grad=True)
        optimizer = torch.optim.SGD([temperature], lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            # Calculate temperature-scaled logits
            scaled_logits = temperature_scale(logits, temperature)
            if self.is_regression:
                true_probabilities = F.softmax(scaled_logits, dim=1)[:, 1]
                true_regressions = true_probabilities * self.max_grade
                loss = self.mse_loss(true_regressions, labels)
            else:
                loss = self.nll_loss(scaled_logits, labels.type(torch.LongTensor))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss.item():.4f}, Temperature: {temperature.item():.4f}"
                )

        return temperature.item()
