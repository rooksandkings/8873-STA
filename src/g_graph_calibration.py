import numpy as np
import pandas as pd
from scipy.integrate import ode, solve_ivp
import random
from itertools import combinations, groupby, cycle
import networkx as nx
from functools import partial
from multiprocessing import Pool
import scipy

# clean list of US states. No US territories.
def get_state_list():
    google_df = pd.read_csv(".\\Data\\Mobility_Data\\Google_Data\\2021_US_Region_Mobility_Report.csv", index_col='date')

    states_us_list = google_df['sub_region_1'].unique()
    states_us_list = [state for state in states_us_list if pd.isnull(state) == False]  # remove null
    states_us_list = [state for state in states_us_list if state != 'District of Columbia']  # remove Washington DC
    return states_us_list

def generate_cliche(connectivity, nodes):
    # code for cliche generation adapted from here:
    # https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx

    probability = connectivity
    G = gnp_random_connected_graph(nodes, probability)

    return G

def simulator(clique_sizes, connectivity, I_0, R_0, beta, gamma, num_sims=2, steps=3):
    # simulator(clique_sizes, .80, I_0 / 10000, R_0 / 10000, beta, gamma, num_sims=5, t_max=7)
    # parameter = [beta, gamma]  # beta, gamma
    # y0 = [1 - I_0 - R_0, I_0, R_0]  # S0, I0, R0
    # Time = np.linspace(0, 30, 900)  # max_time & interval

    # Result = np.zeros([len(Time), 4])
    G = gnp_random_connected_graph(clique_sizes, connectivity)

    NodeList = list(G.nodes())
    N = len(NodeList)
    print('nodes', N)

    I_sim_results, R_sim_results = [], []

    # initialize nodes:
    I_0_nodes = random.choices(NodeList, k=int(I_0))
    R_0_nodes = random.choices(NodeList, k=int(R_0))
    print(len(I_0_nodes), len(R_0_nodes))

    for node in NodeList:
        G.nodes[node]['state'] = 'S'
    for node in I_0_nodes:
        G.nodes[node]['state'] = 'I'
    for node in R_0_nodes:
        G.nodes[node]['state'] = 'R'

    for sim in range(num_sims):
        # step through experiment
        pool = Pool()
        # pool.map(step_through, range(0, 5))
        pool_partial = partial(step_through, G, beta, gamma, steps)
        results = list(pool.map(pool_partial, sim))

    print(results)


def step_through(G, beta, gamma, steps, sim):
    print('sim', sim)
    # step through experiment
    infected_nodes = [x for x, node in G.nodes(data=True) if node['state'] == 'I']
    dead_nodes = [x for x, node in G.nodes(data=True) if node['state'] == 'R']
    print('start infected', len(infected_nodes), 'start deaths', len(dead_nodes))
    for step in range(0, steps):
        print('step', step)
        infected_nodes = [x for x, node in G.nodes(data=True) if node['state'] == 'I']
        for node in infected_nodes:
            if np.random.RandomState().random() < gamma:
                G.nodes[node]['state'] == 'R'
            else:
                NeighborList = [neighbor for neighbor in G.neighbors(node)]
                for neighbor in NeighborList:
                    if G.nodes[neighbor]['state'] == 'S':
                        if np.random.RandomState().random() < beta:
                            G.nodes[neighbor]['state'] = 'I'
    t_infected_nodes = [x for x, node in G.nodes(data=True) if node['state'] == 'I']
    t_dead_nodes = [x for x, node in G.nodes(data=True) if node['state'] == 'R']
    print('end i', len(t_infected_nodes), 'end', len(t_dead_nodes))
    return len(infected_nodes), len(dead_nodes)

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    print('prob', p)
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


if __name__ == "__main__":
    states_us_list = get_state_list()

    states = ['Georgia']
    sources = ['cdc', 'jh']
    time_periods = [7, 14]
    beta_gamma = ['beta', 'gamma']

    test_connectivity_list = np.linspace(80, 90, 2)

    beta = 0.0250000084136493
    gamma = 0.0000802641604853109
    I_0 = 1365436
    R_0 = 21987
    S_0 = 21477737

    clique_sizes = int(S_0 / 10000)
    # for connectivity in test_connectivity_list:
    #     print(connectivity)
    #     forecast_IR(clique_sizes, connectivity, beta, gamma, I_0 / 10000, R_0 / 10000, t_max=7)
    print(I_0 / 10000, R_0 / 10000)

    simulator(clique_sizes, .50, I_0 / 10000, R_0 / 10000, beta, gamma, num_sims=3, steps=7)
