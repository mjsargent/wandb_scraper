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
api = wandb.Api(timeout=30)

sweep_name = "wgzt3439"
#runs = api.runs(path="mjsargent/SRTabular", filters = {"config.max_skips": "7"})
#runs = api.runs(path="mjsargent/SRTabular", filters = {"tags": {"$in": ["hairpinharder_sweep"]}})
#sweep =api.sweep(f"{sweep_name}", filters = {"config.max_skips": "7"})
#"#1AFF1A","#1AFF1A"]
# define sweep parametersp
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
    for agent in ["sq", "tsr"]:
        if not os.path.exists(f"{sweep_name}_{env}_{agent}.pkl"):
            print(f"downloading: {sweep_name}_{env}_{agent}.pkl")
            print(os.path.exists(f"{sweep_name}_{env}_{agent}.pkl"))
            run_names = defaultdict(list)
            run_histories = defaultdict(list)
            env_key = env + "_sweep"
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
for env in envs:
    plt.cla()
    for agent in ["sq", "tsr"]:
        print(agent)
        run_names = []
        history_file = f"{sweep_name}_{env}_{agent}.pkl"
        this_line = Line(run_names = run_names,
                         x_quantity = "_step",
                         y_quantity = "avg_total_temporal_var",
                         color = agent_colour[agent],
                         project = "SRTabular",
                         entity = "mjsargent",
                         local_path = history_file,
                         linestyle = agent_line_style[agent])
        this_line.plot_line(label = agent)
    plt.xlabel("Episodes", fontsize="x-large")
    plt.ylabel("Average Temporal Policy Variation", fontsize="x-large")
    plt.xlim(0,20000)
    plt.ylim(-0.1,1)
    plt.axvline(x=10000, color='grey', linestyle='--')
    plt.title(env_title[env], fontsize="xx-large")
    plt.tight_layout()
    plt.legend(fontsize="x-large", ncol=2)
    plt.grid()
    plt.savefig(f"./figures/Avg_Total_Temporal_Var_{env}_skip_{num_skip}.png", dpi=1200)


