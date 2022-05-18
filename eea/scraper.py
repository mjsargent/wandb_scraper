import wandb
import math
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


class Line:
    def __init__(
        self,
        run_names,
        x_quantity,
        y_quantity,
        points_to_plot=1000000,
        entity="mjsargent",
        project="SRTabular",
        color="blue",
        local_path = None,
        linestyle = "-"
    ):
        self.local_path = local_path
        self.linestyle = linestyle
        self.points_to_plot = points_to_plot
        self.run_names = run_names
        self.api = wandb.Api(timeout = 30)
        self.runs = self.api.runs(entity + "/" + project)
        self.x_quantity = x_quantity
        self.y_quantity = y_quantity
        self.runs = self.grab_runs_based_on_names()
        self.process_run_batch()
        self.color = color

    def process_run_batch(self):
        self.filter_out_nans()
        min_x, max_x = self.get_min_x_max_x()
        x_points = self.sample_x_points(min_x, max_x)
        self.runs = self.make_filtered_runs_have_same_points(x_points)
        mean, std_error = self.compute_mean_and_std_error()
        self.x_points = x_points
        self.mean = mean
        self.std_error = std_error

    def grab_runs_based_on_names(self):
        filtered_runs = []
        if self.local_path == None:
            for a_run in self.runs:
                if a_run.name in self.run_names:
                    history = a_run.history(samples=5 * self.points_to_plot)
                    if self.y_quantity in history.keys():
                        x = history[self.x_quantity].values.tolist()
                        y = history[self.y_quantity].values.tolist()
                        filtered_runs.append([x, y])
        else:
            with open(self.local_path, "rb") as f:
                run_histories = pickle.load(f)

            for set_hist in run_histories.values():
                for history in set_hist:
                    if self.y_quantity in history.keys():
                        x = history[self.x_quantity].values.tolist()
                        y = history[self.y_quantity].values.tolist()
                        filtered_runs.append([x, y])

        return filtered_runs

    def filter_out_nans(self):
        all_filtered_runs = []
        for a_run_xy in self.runs:
            run_without_nan_x = []
            run_without_nan_y = []
            nan = False
            for index, possible_nan in enumerate(a_run_xy[1]):
                if math.isnan(possible_nan) == False:
                    run_without_nan_x.append(a_run_xy[0][index])
                    run_without_nan_y.append(a_run_xy[1][index])
            all_filtered_runs.append([run_without_nan_x, run_without_nan_y])
        self.runs = all_filtered_runs

    def compute_line_equation(self, x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        c = y2 - m * x2
        return m, c

    def sample_x_points(self, min_x, max_x, number_of_points=200):
        interval = (max_x - min_x) / number_of_points
        x_points = [i * interval for i in range(number_of_points)]
        return x_points

    def make_filtered_runs_have_same_points(self, x_points):
        same_point_runs = []
        for a_run in self.runs:
            y_points = []
            for a_x in x_points:
                for index, x_value in enumerate(a_run[0]):
                    if x_value >= a_x:
                        x1 = a_run[0][index - 1]
                        y1 = a_run[1][index - 1]
                        x2 = a_run[0][index]
                        y2 = a_run[1][index]
                        m, c = self.compute_line_equation(x1, y1, x2, y2)
                        y_value = a_x * m + c
                        y_points.append(y_value)
                        break
            assert len(y_points) == len(x_points)
            same_point_runs.append([x_points, y_points])
        return same_point_runs

    def compute_mean_and_std_error(self):
        mean_list = []
        std_error_list = []
        for index, a_x_point in enumerate(self.runs[0][0]):
            points = []
            for a_run in self.runs:
                points.append(a_run[1][index])
            points = np.array(points)
            mean_list.append(np.mean(points))
            std_error_list.append(stats.sem(points))
        return mean_list, std_error_list

    def get_min_x_max_x(self):
        min_x = max([a_run[0][0] for a_run in self.runs])
        max_x = min([a_run[0][-1] for a_run in self.runs])
        return min_x, max_x

    def plot_line(self, ax,  color=None, alpha=None, label=None, inset_axis = None, limits: list = []):
        if color == None:
            color = self.color
        if alpha == None:
            alpha = 0.2
        alpha_plot = 0.6
        ax.plot(self.x_points, self.mean, color=color, label=label, alpha =  alpha_plot, linestyle = self.linestyle, linewidth = 2.5)
        ax.fill_between(
            self.x_points,
            np.array(self.mean) - np.array(self.std_error),
            np.array(self.mean) + np.array(self.std_error),
            alpha=alpha,
            color=color,
        )

        if inset_axis !=  None:
            inset_axis.plot(self.x_points, self.mean, color=color, label=label, alpha =  alpha_plot, linestyle = self.linestyle, linewidth = 2.5)
            inset_axis.fill_between(
            self.x_points,
            np.array(self.mean) - np.array(self.std_error),
            np.array(self.mean) + np.array(self.std_error),
            alpha=alpha,
            color=color,
            )
            inset_axis.plot
            # sub region of the original image
            x1, x2, y1, y2 = limits
            #x1, x2, y1, y2 = 9500, 10999, 0, 1.1
            inset_axis.set_xlim(x1, x2)
            inset_axis.set_ylim(y1, y2)
            inset_axis.xaxis.set_tick_params(labelsize=8)
            inset_axis.yaxis.set_tick_params(labelsize=8)
           # inset_axis.set_xticks([])
           # inset_axis.set_yticks([])
            rect, connect = ax.indicate_inset_zoom(inset_axis, edgecolor="black", label='_nolegend_')
            connect[0].set_visible(False)
            connect[1].set_visible(False)
            connect[2].set_visible(False)
            connect[3].set_visible(False)

        return ax

def main():
    line_1 = Line(
        run_names=[
            "blooming-glade-1549",
            "drawn-pond-1551",
            "lyric-firefly-1553",
            "comfy-pond-1555",
            "breezy-frog-1557",
        ],
        x_quantity="_step",
        y_quantity="fixed visiation counts",
        color="blue",
    )
    line_1.plot_line()
    plt.show()


if __name__ == "__main__":
    main()
