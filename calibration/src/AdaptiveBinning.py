# from __future__ import division
import matplotlib.pyplot as plt


def AdaptiveBinning(infer_results, show_reliability_diagram=True):
    """
    This function implement adaptive binning. It returns AECE, AMCE and some other useful values.

    Arguements:
    infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample. res[0] is the confidence score r and res[1] is the correctness score c. Since c is either 1 or 0, here res[1] is True if the prediction is correctd and False otherwise.
    show_reliability_diagram (boolean): a boolean value to denote wheather to plot a Reliability Diagram.

    Return Values:
    AECE (float): expected calibration error based on adaptive binning.
    AMCE (float): maximum calibration error based on adaptive binning.
    cofidence (list): average confidence in each bin.
    accuracy (list): average accuracy in each bin.
    cof_min (list): minimum of confidence in each bin.
    cof_max (list): maximum of confidence in each bin.

    """

    # Intialize.
    infer_results.sort(key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    # assert (
    #     infer_results[0][0] <= 1 and infer_results[1][0] >= 0
    # ), "Confidence score should be in [0,1]"

    z = 1.645
    num = [0 for i in range(n_total_sample)]
    final_num = [0 for i in range(n_total_sample)]
    correct = [0 for i in range(n_total_sample)]
    confidence = [0 for i in range(n_total_sample)]
    cof_min = [1 for i in range(n_total_sample)]
    cof_max = [0 for i in range(n_total_sample)]
    accuracy = [0 for i in range(n_total_sample)]

    ind = 0
    target_number_samples = float("inf")

    # Traverse all samples for a initial binning.
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 40 and cof_min[ind] - infer_results[-1][0] > 0.05:
                ind += 1
                target_number_samples = float("inf")
        num[ind] += 1
        confidence[ind] += confidence_score

        # assert correctness in [True, False], "Expect boolean value for correctness!"
        # if correctness == True:
        #     correct[ind] += 1

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float("inf")
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    print([x for x in num if x != 0])
    # assert False

    n_bins = ind + 1

    # Get final binning.
    if target_number_samples - num[ind] > 0:
        needed = target_number_samples - num[ind]
        extract = [0 for i in range(n_bins - 1)]
        final_num[n_bins - 1] = num[n_bins - 1]
        for i in range(n_bins - 1):
            extract[i] = int(needed * num[ind] / n_total_sample)
            final_num[i] = num[i] - extract[i]
            final_num[n_bins - 1] += extract[i]
    else:
        final_num = num
    final_num = final_num[:n_bins]

    # Re-intialize.
    num = [0 for i in range(n_bins)]
    correct = [0 for i in range(n_bins)]
    confidence = [0 for i in range(n_bins)]
    cof_min = [1 for i in range(n_bins)]
    cof_max = [0 for i in range(n_bins)]
    accuracy = [0 for i in range(n_bins)]
    gap = [0 for i in range(n_bins)]
    neg_gap = [0 for i in range(n_bins)]
    # Bar location and width.
    x_location = [0 for i in range(n_bins)]
    width = [0 for i in range(n_bins)]

    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        num[ind] += 1
        confidence[ind] += confidence_score

        if correctness == True:
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - accuracy[ind] > 0:
                gap[ind] = confidence[ind] - accuracy[ind]
            else:
                neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    # Get AECE and AMCE based on the binning.
    AMCE = 0
    AECE = 0
    for i in range(n_bins):
        AECE += abs((accuracy[i] - confidence[i])) * final_num[i] / n_total_sample
        AMCE = max(AMCE, abs((accuracy[i] - confidence[i])))

    # Plot the Reliability Diagram if needed.
    if show_reliability_diagram:
        f1, ax = plt.subplots()
        plt.bar(x_location, accuracy, width)
        plt.bar(x_location, gap, width, bottom=accuracy)
        plt.bar(x_location, neg_gap, width, bottom=accuracy)
        plt.legend(["Accuracy", "Positive gap", "Negative gap"], fontsize=18, loc=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Confidence", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.show()

    return AECE, AMCE, cof_min, cof_max, confidence, accuracy



def AdaptiveBinningForRegression(infer_results, show_reliability_diagram=True):
    """
    This function implement adaptive binning. It returns AECE, AMCE and some other useful values.

    Arguements:
    infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample.
    show_reliability_diagram (boolean): a boolean value to denote wheather to plot a Reliability Diagram.

    Return Values:
    AECE (float): expected calibration error based on adaptive binning.
    AMCE (float): maximum calibration error based on adaptive binning.
    cofidence (list): average confidence in each bin.
    accuracy (list): average accuracy in each bin.
    cof_min (list): minimum of confidence in each bin.
    cof_max (list): maximum of confidence in each bin.

    """

    # Intialize.
    infer_results.sort(key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    max_score = infer_results[0][0]

    z = 1.645
    num = [0 for _ in range(n_total_sample)]
    final_num = [0 for _ in range(n_total_sample)]
    correct = [0 for _ in range(n_total_sample)]
    confidence = [0 for _ in range(n_total_sample)]
    cof_min = [max_score for _ in range(n_total_sample)]
    cof_max = [0 for _ in range(n_total_sample)]
    accuracy = [0 for _ in range(n_total_sample)]

    ind = 0
    target_number_samples = float("inf")

    # Traverse all samples for a initial binning.
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 10 and cof_min[ind] - infer_results[-1][0] > 0.01:
                ind += 1
                target_number_samples = float("inf")
        num[ind] += 1
        confidence[ind] += confidence_score

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float("inf")
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    n_bins = ind + 1

    # print(target_number_samples, num[ind])



    # print(n_bins, [x for x in num if x != 0])
    # print(cof_min, cof_max)
    # assert False

    # # Get final binning.
    # if target_number_samples - num[ind] > 0:
    #     needed = target_number_samples - num[ind]
    #     extract = [0 for _ in range(n_bins - 1)]
    #     final_num[n_bins - 1] = num[n_bins - 1]
    #     for i in range(n_bins - 1):
    #         extract[i] = int(needed * num[ind] / n_total_sample)
    #         final_num[i] = num[i] - extract[i]
    #         final_num[n_bins - 1] += extract[i]
    # else:
    #     final_num = num
    # final_num = final_num[:n_bins]

    final_num = num[:n_bins]

    # print(final_num)

    # Re-intialize.
    num = [0 for _ in range(n_bins)]
    correct = [0 for _ in range(n_bins)]
    confidence = [0 for _ in range(n_bins)]
    cof_min = [max_score for _ in range(n_bins)]
    cof_max = [0 for _ in range(n_bins)]
    accuracy = [0 for _ in range(n_bins)]
    gap = [0 for _ in range(n_bins)]
    neg_gap = [0 for _ in range(n_bins)]
    # Bar location and width.
    x_location = [0 for _ in range(n_bins)]
    width = [0 for _ in range(n_bins)]

    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        num[ind] += 1
        confidence[ind] += confidence_score

        # if correctness == True:
        #     correct[ind] += 1
        correct[ind] += correctness
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - accuracy[ind] > 0:
                gap[ind] = confidence[ind] - accuracy[ind]
            else:
                neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    # Get AECE and AMCE based on the binning.
    AMCE = 0
    AECE = 0
    aece_per_bin = []
    for i in range(n_bins):
        aece_this_bin = abs((accuracy[i] - confidence[i])) * final_num[i] / n_total_sample
        AECE += aece_this_bin
        aece_per_bin.append(aece_this_bin)
        AMCE = max(AMCE, abs((accuracy[i] - confidence[i])))

    # Plot the Reliability Diagram if needed.
    if show_reliability_diagram:
        _, ax1 = plt.subplots(figsize=(6, 6))
        ax1.plot(range(0, 4), range(0, 4), label="ideal", linestyle="--", color="black")
        ax1.bar(x_location, accuracy, width, label="accuracy", color="grey")
        ax1.bar(x_location, gap, width, bottom=accuracy, label="positive gap", color="red")
        ax1.bar(x_location, neg_gap, width, bottom=accuracy, label="negative gap", color="green")
        
        ax1.set_xlim(0, max_score)
        ax1.set_ylim(0, max_score)
        ax1.set_xlabel("Confidence", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylim(0, 0.15)
        ax2.set_ylabel("ECE per bin", color="blue")
        ax2.plot(x_location, aece_per_bin, marker='.', color='b', markerfacecolor='none', markeredgecolor='b', ms=10)
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax1.legend(fontsize=12, loc=2)
        plt.title(f"Reliability Diagram -- {n_bins} bins\n AECE={AECE:.3f}", fontsize=15)
        plt.show()

    return AECE, AMCE, cof_min, cof_max, confidence, accuracy, aece_per_bin
