"""calculates 2d error of model on STIR labelled dataset."""

from datetime import datetime
import cv2
import numpy as np
import json
import sys
from tqdm import tqdm
from collections import defaultdict
import itertools
import os

from STIRLoader import STIRLoader
import random
import torch
import argparse
from scipy.spatial import KDTree
from pathlib import Path
import logging
from testutil import *


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--startgt",
        type=str,
        default="",
        help="output suffix for json",
    )
    parser.add_argument(
        "--endgt",
        type=str,
        default="",
        help="output suffix for json",
    )
    parser.add_argument(
        "--model_predictions",
        type=str,
        default="",
        help="model predictions json",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="directory with STIR Challenge 2024 dataset",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/stirc_2024",
        help="directory to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LiteTracker",
        help="model",
    )

    args = parser.parse_args()
    args.batch_size = 1  # do not change, only one for running
    return args


def calculate_accuracy(f, distances):
    thresholds = [4, 8, 16, 32, 64]
    distances = np.array(distances)
    num_samples = len(distances)
    accuracies = []
    for threshold in thresholds:
        number_below_thresh = np.sum(distances <= threshold)
        accuracy_at_thresh = number_below_thresh / num_samples
        accuracies.append(accuracy_at_thresh)
        f.write(f"{threshold:2d} px:\t{accuracy_at_thresh:0.5f}\n")
        print(f"{threshold:2d} px:\t{accuracy_at_thresh:0.5f}")

    avg_accuracy = np.mean(accuracies)
    print(f"Avg:\t{avg_accuracy:0.5f}")
    f.write(f"Avg:\t{avg_accuracy:0.5f}\n")


if __name__ == "__main__":
    args = getargs()
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    with open(args.model_predictions, "r") as f:
        model_prediction_dict = json.load(f)
    with open(args.startgt, "r") as f:
        start_gt_dict = json.load(f)
    with open(args.endgt, "r") as f:
        end_gt_dict = json.load(f)
    errors_control_avg = 0.0
    errors_avg = defaultdict(int)
    data_used_count = 0
    distances = []
    control_distances = []
    for filename, pointlist_model in model_prediction_dict.items():
        assert filename in start_gt_dict
        assert filename in end_gt_dict
        pointlist_model = np.array(pointlist_model)
        pointlist_start = np.array(start_gt_dict[filename])
        pointlist_end = np.array(end_gt_dict[filename])

        errors_control = pointlossunidirectional(pointlist_start, pointlist_end)
        control_distancelist = errors_control["distancelist"]
        control_distances.extend(control_distancelist)

        errors = pointlossunidirectional(pointlist_model, pointlist_end)
        distancelist = errors["distancelist"]
        distances.extend(distancelist)
        data_used_count += 1
        print(f"Calculated distances for: {filename}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f = open(
        os.path.join(args.results_dir, f"{timestamp}_{args.model}.log"), "w"
    )
    f.write("CONTROL\n")
    f.write("-----\n")
    f.write("Accuracy\n")
    f.write("-----\n")
    print("CONTROL")
    print("-----")
    print("Accuracy")
    print("-----")
    calculate_accuracy(f, control_distances)
    print("----------------")
    print("Model")
    print("-----")
    print("Accuracy")
    print("-----")
    f.write("----------------\n")
    f.write("Model\n")
    f.write("-----\n")
    f.write("Accuracy\n")
    f.write("-----\n")
    calculate_accuracy(f, distances)
