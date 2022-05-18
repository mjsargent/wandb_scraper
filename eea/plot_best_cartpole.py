import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scraper import Line
from collections import defaultdict

#plot settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')

def legend_without_duplicate_labels(ax, agents):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if (l not in labels[:i] and l in agents)]

    ax.legend(*zip(*unique), fontsize = "x-large", ncol = 2)

####
#  _step is the number of total updates (number of times "wandb.log" is called)
####

### Quantities
## DQN:
# return
# episode
# episode_length
# total_steps
#
## A2C:
# ReturnAverage
# LengthAverage
# CumuCompletedTrajs (episodes)
# CumuSteps (steps in env)
#
###


# NB - does not change the axis labels or the file name!
x_quantity_a2c = "_step"
y_quantity_a2c = "ReturnAverage"
x_quantity_dqn = "_step"
y_quantity_dqn = "return"



env = "cartpole"
# define sweep parametersp
#colours = ["#AA4499", "#CC6677", "#DDCC77", "#117733", "#88CCEE"]
colours = ["#1A85FF","#000000",  "#1AFF1A", "#D41159", "#5D3A9B"]
#titles = ["Junction", "Junction Hard", "Junction Very Hard"]
linestyles = ["--", "--", "-", "-.", "-"]
agent_colour = {}
agent_line_style = {}

#wandb init
api = wandb.Api(timeout=60)

agents = ["equivariant_a2c", "mlp_a2c"]
tags = ["best_equi_a2c", "best_mlp_a2c"]

x_quantity_dict = {}
y_quantity_dict = {}



tags_dict = {}
for tag, agent in zip(tags, agents):
    tags_dict[agent] = tag
    x_quantity_dict[agent] = x_quantity_a2c
    y_quantity_dict[agent] = y_quantity_a2c


# pred prey results are split over two locations (oops) so we will loop
# through them seperately

for agent in agents:
    if not os.path.exists(f"{env}_{agent}.pkl"):
            print(f"downloading: {env}_{agent}")
            run_names = defaultdict(list)
            run_histories = defaultdict(list)
            env_key = tags_dict[agent]
            runs = api.runs(path="barry_lab/cartpole", filters = {"tags": {"$in": [env_key]}})
            for i, run in enumerate(runs):
                print("processing run", i)

                run_histories[i].append(run.history(samples = 100000))

            with open(f"{env}_{agent}.pkl", "wb") as f:
                pickle.dump(run_histories, f)

            print(f"pickled: {env}_{agent}.pkl")

# TODO loop through the runs stuck in cartpole

agents_dqn = ["equivariant_dqn", "nonlinear_dqn", "linear_dqn"]
tags_dqn = ["best_equi_dqn", "best_nonlinear_dqn", "best_linear_dqn"]


tags_dict = {}
for tag, agent in zip(tags_dqn, agents_dqn):
    tags_dict[agent] = tag
    x_quantity_dict[agent] = x_quantity_dqn
    y_quantity_dict[agent] = y_quantity_dqn

for agent in agents_dqn:

    if not os.path.exists(f"{env}_{agent}.pkl"):
            print(f"downloading: {env}_{agent}")
            run_names = defaultdict(list)
            run_histories = defaultdict(list)
            env_key = tags_dict[agent]
            runs = api.runs(path="barry_lab/cartpole", filters = {"tags": {"$in": [env_key]}})
            for i, run in enumerate(runs):
                print("processing run", i)

                run_histories[i].append(run.history(samples = 100000))

            with open(f"{env}_{agent}.pkl", "wb") as f:
                pickle.dump(run_histories, f)

            print(f"pickled: {env}_{agent}.pkl")


agents = agents + agents_dqn

inset_axis = None
fig, ax = plt.subplots()
ax.clear()

for i, agent in enumerate(agents):
    agent_colour[agent] = colours[i]
    agent_line_style[agent] = linestyles[i]


### PLOTTING

for agent in agents:

    print(agent)
    run_names = []
    history_file = f"{env}_{agent}.pkl"
    x_q = x_quantity_dict[agent]
    y_q = y_quantity_dict[agent]
    this_line = Line(run_names = run_names,
                        x_quantity = x_q,
                        y_quantity = y_q,
                        color = agent_colour[agent],
                        project = "cartpole",
                        entity = "barry_lab",
                        local_path = history_file,
                        linestyle = agent_line_style[agent])
    ax = this_line.plot_line(label = agent, ax = ax, inset_axis = inset_axis, limits = None)
plt.xlabel("Cummulative Steps", fontsize="x-large")
plt.ylabel("Average Return", fontsize="x-large")
plt.xlim(0,200)
plt.ylim(0, 505)
plt.title(env, fontsize="xx-large")
plt.tight_layout()
plt.legend(agents)
legend_without_duplicate_labels(ax, agents)
#plt.legend(fontsize="x-large", ncol=3, loc = "lower center")
plt.legend(fontsize="x-large", ncol = 1)
#if a == 2:

plt.grid()
plt.savefig(f"./figures/{env}_average_return.png", dpi=600)
#if a == 2:
#    print(inset_axis.patches)
#jj    for p in reversed(list(inset_axis.patches)):    # note the list!
    #       p.set_visible(False)
    #       p.remove()

