import networkx as nx
#import ndlib.models.ModelConfig as mc
#import ndlib.models.epidemics as ep
import models.ModelConfig as mc
import models.epidemics as ep
import numpy as np
from pprint import pprint

np.random.seed(0)


def runIC(network, seed_set, k, config, MC):
    model = ep.IndependentCascadesModel(network)
    config.add_model_initial_configuration("Infected", seed_set)
    model.set_initial_status(config)

    probs = np.zeros(network.number_of_nodes())

    edge_probs = config.get_edges_configuration()['threshold']
    joint_pdfs = []
    # Per round or per realization gamma values which we will sum later.
    gamma = {}

    print("About to run our simulation rounds")

    for i in range(MC):
        if i % 1000 == 0:
            print(f"On round {i+1}")

        gamma_X = dict.fromkeys(network.nodes, 0)

        model.reset()
        model.set_initial_status(config)
        iterations = model.iteration_bunch(nx.diameter(network))

        running_prod = 1
        for edge, val in iterations[-1]['edge_status'].items():
            if val == 1:
                running_prod *= edge_probs[edge]
            else:
                running_prod *= (1 - edge_probs[edge])

        joint_pdfs.append(running_prod)
        for node, val in iterations[-1]["node_infection_status"].items():
            gamma_X[node] = running_prod * val            
        
        gamma[i] = gamma_X

    print("Completed simulation rounds, now summing up gamma values")
    # Get sum over X, sum over gamma_X with mult from joint_pdfs
    for n in network.nodes:
        for round in range(MC):
            probs[n] += gamma[round][n]

    print("Now calculating F values that we need")
    F = 0
    argprobs = np.argsort(probs)
    cnts = 0
    for idx in argprobs[::-1]:
        if idx not in seed_set and cnts < k:
            cnts += 1
            F += probs[idx]
            if cnts == k:
                break

    return F


if __name__ == "__main__":
    # G = nx.erdos_renyi_graph(50, 0.5)
    G = nx.karate_club_graph()
    # G = nx.bull_graph()
    # G = nx.barabasi_albert_graph(50, 10)
    G = G.to_undirected()
    N = G.number_of_nodes()
    E = G.number_of_edges()
    k = 5
    S = list(np.random.choice(range(N), size=4, replace=False))
    T = []
    while len(T) <= 4:
        node = list(np.random.choice(range(N), size=1, replace=False))
        if node not in S:
            T.append(node[0])
    T = S + T

    v = np.random.choice(range(N))
    while v in T:
        v = np.random.choice(range(N))

    config = mc.Configuration()

    for e in G.edges():
        config.add_edge_configuration(
            "threshold", e, 0.5)

    MC = 10000

    F_S = runIC(G, S, k, config, MC)
    F_S_v = runIC(G, S + [v], k, config, MC)
    F_T = runIC(G, T, k, config, MC)
    F_T_v = runIC(G, T + [v], k, config, MC)

    pprint(f"F(S) = {F_S}")
    pprint(f"F(S U v) = {F_S_v}")
    pprint(f"F(T) = {F_T}")
    pprint(f"F(T U v) = {F_T_v}")
    pprint(f"F(S U v) - F(S) = {F_S_v - F_S}")
    pprint(f"F(T U v) - F(T) = {F_T_v - F_T}")

    F_T = runIC(G, T, k, config, MC)
    F_T_v = runIC(G, T + [v], k, config, MC)

    pprint(f"F(S) = {F_S}")
    pprint(f"F(S U v) = {F_S_v}")
    pprint(f"F(T) = {F_T}")
    pprint(f"F(T U v) = {F_T_v}")
    pprint(f"F(S U v) - F(S) = {F_S_v - F_S}")
    pprint(f"F(T U v) - F(T) = {F_T_v - F_T}")

