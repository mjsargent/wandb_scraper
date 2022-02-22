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

#wandb init
api = wandb.Api(timeout=60)

# define sweep name if
sweep_name = "71s9i8je"

# define sweep parameters
agents = ["q", "sq", "sr", "sr_extend", "tsr"]
#colours = ["#AA4499", "#CC6677", "#DDCC77", "#117733", "#88CCEE"]
colours = ["#1A85FF","#000000",  "#1AFF1A", "#D41159", "#5D3A9B"]
titles = ["Junction", "Junction Hard", "Junction Very Hard"]
linestyles = ["--", "--", "-", "-.", "-"]
agent_colour = {}
agent_line_style = {}
for i, agent in enumerate(agents):
    agent_colour[agent] = colours[i]
    agent_line_style[agent] = linestyles[i]


envs = ["hairpin", "hairpinhard", "hairpinharder"]
env_title = {}
for i, env in enumerate(envs):
    env_title[env] = titles[i]

num_skips = [7]

# group runs by agent, env and skip
# save this locally if it does not exist

# loop through three times, but save on memory
for env in envs:
    #for agent in agents:
    for agent in agents:
        if not os.path.exists(f"{sweep_name}_{env}_{agent}.pkl"):
            print(f"downloading: {sweep_name}_{env}_{agent}.pkl")
            run_names = defaultdict(list)
            run_histories = defaultdict(list)
            env_key = env + "_sweep"

            # this is the key line to edit to change the runs you are grabbing
            runs = api.runs(path="mjsargent/SRTabular", filters = {"tags": {"$in": [env_key]}, "config.agent": agent})

            for i, run in enumerate(runs):
                print("processing run", i)
                key = (run.config["agent"], run.config["env"], run.config["max_skips"])
                run_names[key].append(run.name)
                run_histories[key].append(run.history(samples = 100000))

            with open(f"{sweep_name}_{env}_{agent}.pkl", "wb") as f:
                pickle.dump(run_histories, f)

            print(f"pickled: {sweep_name}_{env}_{agent}.pkl")
#  for num_skip in num_skips: - ignore looping this for now
num_skip = num_skips[0]
limits = [9500, 10999, -0.01, 1]
# plot backwards because of some oddities with the inset axes zoom
for a, env in enumerate(reversed(envs)):
    inset_axis = None
    fig, ax = plt.subplots()
    ax.clear()

    plot_inset = True if a == 2 else False
    if plot_inset:
        inset_axis = ax.inset_axes([12750, -0.25, 7250, 1 ], transform = ax.transData)
    for agent in agents:

        print(agent)
        run_names = []
        history_file = f"{sweep_name}_{env}_{agent}.pkl"

        this_line = Line(run_names = run_names,
                         x_quantity = "_step",
                         y_quantity = "test_episode_reward:",
                         color = agent_colour[agent],
                         project = "SRTabular",
                         entity = "mjsargent",
                         local_path = history_file,
                         linestyle = agent_line_style[agent])
        ax = this_line.plot_line(label = agent, ax = ax, inset_axis = inset_axis, limits = limits)
    plt.xlabel("Episodes", fontsize="x-large")
    plt.ylabel("Average Episode Return", fontsize="x-large")
    plt.xlim(0,19999)
    plt.ylim(-1,1.2)
    plt.axvline(x=10000, color='grey', linestyle='--')
    plt.title(env_title[env], fontsize="xx-large")
    plt.tight_layout()
    if a == 2:
        #legend_without_duplicate_labels(ax, agents)

        plt.legend(fontsize="x-large", ncol=2)

    plt.grid()
    plt.savefig(f"./figures/Test_Episode_Reward_{env}_skip_{num_skip}.png", dpi=1200)
    if a == 2:
        print(inset_axis.patches)
        for p in reversed(list(inset_axis.patches)):    # note the list!
            p.set_visible(False)
            p.remove()



