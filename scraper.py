import pandas as pd
import wandb
import math
import matplotlib.pyplot as plt 
from scipy import stats
import numpy as np

def grab_runs_based_on_names(runs, name_list, x_name, y_name):
    filtered_runs = []
    for a_run in runs:
        if a_run.name in name_list:
            history = a_run.history()
            x = history[x_name].values.tolist()
            y = history[y_name].values.tolist()
            filtered_runs.append([x, y])
    return filtered_runs

def filter_out_nans(run_list):
    all_filtered_runs = []
    for a_run_xy in run_list:
        run_without_nan_x = []
        run_without_nan_y = []
        nan = False
        for index, possible_nan in enumerate(a_run_xy[1]):
            if math.isnan(possible_nan) == False:
                run_without_nan_x.append(a_run_xy[0][index])
                run_without_nan_y.append(a_run_xy[1][index])
        all_filtered_runs.append([run_without_nan_x, run_without_nan_y])
    return all_filtered_runs

def compute_line_equation(x1, y1, x2, y2):
    m = (y2 - y1)/(x2 - x1)
    c = y2 - m * x2 
    return m, c

def compute_y_value(x, m, c):
    return m * x + c

def sample_x_points(a_filtered_run, number_of_points = 1000):
    max_x = max(a_filtered_run)
    min_x = min(a_filtered_run)
    interval = (max_x - min_x)/number_of_points
    x_points = [i * interval for i in range(number_of_points)]
    return x_points

def make_filtered_runs_have_same_points(runs, x_points):
    same_point_runs = []
    for a_run in runs:     
        y_points = []
        for a_x in x_points:
            for index, x_value in enumerate(a_run[0]):
                if x_value > a_x: 
                    x1 = a_run[0][index - 1] 
                    y1 = a_run[1][index - 1]
                    x2 = a_run[0][index] 
                    y2 = a_run[1][index]
                    m, c = compute_line_equation(x1, y1, x2, y2)
                    y_value = a_x * m + c
                    y_points.append(y_value)
                    break
        same_point_runs.append([x_points, y_points])
    return same_point_runs

def compute_mean_and_std_error(runs):
    mean_list = []
    std_error_list = []
    for index, a_x_point in enumerate(runs[0][0]):
        points = []
        for a_run in runs:
            try:
                points.append(a_run[1][index])
            except:
                import ipdb; ipdb.set_trace() 
        points = np.array(points)
        mean_list.append(np.mean(points))
        std_error_list.append(stats.sem(points))
    return mean_list, std_error_list

def process_run_batch(runs):
    runs = filter_out_nans(runs)
    x_points = sample_x_points(runs[0][0])
    runs = make_filtered_runs_have_same_points(runs, x_points) 
    mean, std_error = compute_mean_and_std_error(runs) 
    return x_points, mean, std_error

def main():
    api = wandb.Api()
    entity, project = "self-supervisor", "minigrid"  # set to your entity and project
    runs = api.runs(entity + "/" + project)
    curious_true_uncertainty_true_noisy_true_4_rooms = [
        "blooming-glade-1549",
        "drawn-pond-1551",
        "lyric-firefly-1553",
        "comfy-pond-1555",
        "breezy-frog-1557",
    ]
    curious_true_uncertainty_true_noisy_true_4_rooms = grab_runs_based_on_names(
        runs=runs,
        name_list=curious_true_uncertainty_true_noisy_true_4_rooms,
        x_name="_step",
        y_name="fixed visiation counts",
    )
    x_points, mean, std_error = process_run_batch(curious_true_uncertainty_true_noisy_true_4_rooms)
    plt.plot(x_points, mean)
    plt.show()

if __name__ == "__main__":
    main()
