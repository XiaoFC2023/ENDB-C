
import math
import random
import time
from itertools import combinations
from numbers import Integral

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gurobipy import GRB

from NDSG import New#You may substitute other high-quality heuristic methods and replace `New` accordingly.


def draw(G1,remove_nodes,title):
    
    plt.figure(figsize=(6, 6))
    
    G=G1.copy(as_view=0)
    
    pos=nx.spring_layout(G,seed=1)
    

    node_degrees = dict(G.degree())
    node_sizes = [v ** 2.5+50 for v in node_degrees.values()]
    
    nx.draw(G, pos=pos, with_labels=0, node_color='w', edge_color='k', node_size=100)
    nx.draw_networkx_nodes(G,  pos=pos, node_color=[0.8, 0.8, 0.8], node_size=100)
    
    
    remove_node_sizes = [v ** 2.5+50 for v in node_degrees.values()]
    for u in set(list(G1.nodes))-set(remove_nodes):
        remove_node_sizes[u] = 0
        
    nx.draw(G.subgraph(remove_nodes), pos=pos, node_color=[227/255,5/255,41/255], node_size=100)
    # plt.title(f'{title},remove nodes:{len(remove_nodes)}')
    
    G.remove_nodes_from(remove_nodes)
    # nodes=list(G.nodes)
    # G=nx.convert_node_labels_to_integers(G)
    
    L = list(nx.connected_components(G))
    max_L = max(L, key=lambda x: len(x))
    
    
    plt.show()
    
    plt.figure(figsize=(6, 6))
    
    nx.draw(G, pos=pos, with_labels=0, node_color=[0.8, 0.8, 0.8], edge_color='k', node_size=200)
    nx.draw_networkx_nodes(G,  pos=pos, node_color=[0.8, 0.8, 0.8], node_size=200)
    
    max_Ls=[i for i in max_L]
    for c in L:
        if len(c)==len(max_L):
            max_Ls.extend(list(c))
    nx.draw(G.subgraph(max_L), pos=pos, node_color='b', node_size=200)
    # plt.title(f'LCC:{len(max_L)}')
    print(f'LCC:{len(max_L)}')

    plt.show()



def correct_remove_nodes(G,alpha_n,remove_nodes):

    nodes=set(list(G.nodes))
    remain_nodes=nodes-set(remove_nodes)
    correct_remove_nodes=[]
    for remove_node in remove_nodes:
        nei_remove_nodes=set(list(G[remove_node]))
        
        if len(nei_remove_nodes & remain_nodes)==0:
            correct_remove_nodes.append(remove_node)
    

    if len(correct_remove_nodes)<=alpha_n:
        remove_nodes=list(set(remove_nodes)-set(correct_remove_nodes))
        
        return remove_nodes
        

    G_remove=G.subgraph(remove_nodes)
    origial_nodes=list(G_remove)
    # print(origial_nodes)
    G_remove=nx.convert_node_labels_to_integers(G_remove, first_label=0, ordering='default', label_attribute=None)
    

    V=list(G_remove.nodes)
    N=len(V)
    K=int(np.ceil(len(correct_remove_nodes)/alpha_n))
    # print(len(correct_remove_nodes))
    
    model = gp.Model()
    X=model.addVars(N,K+1,vtype=GRB.BINARY)
    

    for node in V:
        if origial_nodes[node] not in correct_remove_nodes:
            model.addConstr(X[node,0]==1)

    for node in V:
        for k in range(1,K+1):
            for j in list(G_remove[node]):
                model.addConstr(X[node,k]<=X[j,k]+X[j,0])

    for node in V:
        if origial_nodes[node] in correct_remove_nodes:
            model.addConstr(X[node,0]<=gp.quicksum(1-X[j,0] for j in list(G_remove[node])))
    for k in range(1,K+1):
        model.addConstr(gp.quicksum(X[i,k] for i in V)<=alpha_n)
    for i in V:
        model.addConstr(gp.quicksum(X[i,k] for k in range(K+1))==1)
    
    model.setObjective(gp.quicksum(X[i,0] for i in V), GRB.MINIMIZE)
    
    model.setParam("OutputFlag", 0)
    
    model.optimize()
    
    optimal_x = np.array([X[i,0].x for i in V])
    deleted = np.where(optimal_x >= 0.99)[0]
    
    optimal_remove_nodes=[]
    for i in deleted:
        optimal_remove_nodes.append(origial_nodes[i])
    return optimal_remove_nodes


def _degree_lower_bound(G, alpha_n, upper_bound):
    """Apply Eq. (26) to a connected graph and a feasible upper bound."""
    n = len(G)
    target = (n - upper_bound + alpha_n - 1) // alpha_n
    if target == 0:
        return 0
    degrees = sorted((degree for _, degree in G.degree()), reverse=True)
    cumulative = 0
    for k, degree in enumerate(degrees, start=1):
        cumulative += degree
        if cumulative >= target:
            return k
    raise ValueError("Eq. (26) requires a connected graph and a valid upper bound.")


def SIMPLR_LB(G, alpha_n, r):
    """Evaluate the degree bound using the feasible removal set r."""
    if len(G) <= alpha_n:
        return 0
    return _degree_lower_bound(G, alpha_n, len(r))


def _component_cuts(G, component, alpha_n, rng):
    """Generate Algorithm 1 cuts from one oversized connected component."""
    size = alpha_n + 1
    nodes = sorted(component)
    if len(nodes) < size:
        return
    if len(nodes) == size:
        yield tuple(nodes)
        return

    H = G.subgraph(nodes)
    if len(nodes) <= alpha_n + 3:
        for cut in combinations(nodes, size):
            if nx.is_connected(H.subgraph(cut)):
                yield cut
        return

    neighbors = {node: tuple(sorted(H[node])) for node in nodes}
    uncovered = set(nodes)
    while uncovered:
        node = rng.choice(sorted(uncovered))
        visited = {node}
        while len(visited) < size:
            node = rng.choice(neighbors[node])
            visited.add(node)
        uncovered.difference_update(visited)
        yield tuple(sorted(visited))


def _restore_candidate(G, alpha_n, removed, leaf_neighbors):
    """Prepare a feasible candidate that satisfies the enabled master constraints."""
    removed = set(removed)
    # Exchange a removed leaf with its neighbor before calling the original
    # restoration routine. This preserves feasibility and cannot increase |S|.
    for leaf, neighbor in leaf_neighbors.items():
        if leaf in removed:
            removed.remove(leaf)
            removed.add(neighbor)
    return frozenset(correct_remove_nodes(G, alpha_n, sorted(removed)))


def _integer_solver_bound(bound):
    """Round a finite global bound conservatively for the integer objective."""
    if not math.isfinite(bound) or abs(bound) >= GRB.INFINITY:
        return None
    return math.ceil(bound - 1e-6)


def _update_solver_bound(model, bound):
    integer_bound = _integer_solver_bound(bound)
    if integer_bound is not None:
        model._lb = max(model._lb, integer_bound)
    return integer_bound


def my_callback(model, where):
    x = model._x_vars
    if where == GRB.Callback.MIP:
        _update_solver_bound(model, model.cbGet(GRB.Callback.MIP_OBJBND))
        if model._lb == len(model._best) or time.perf_counter() >= model._deadline:
            model.terminate()
        return

    if where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            bound = _update_solver_bound(model, model.cbGet(GRB.Callback.MIPNODE_OBJBND))
            if bound is not None and bound > model._objective_cut_bound:
                # Eq. (40) uses the global bound, so the cut can be added globally.
                model.cbCut(model._objective >= bound)
                model._objective_cut_bound = bound
        return

    if where != GRB.Callback.MIPSOL:
        return

    G = model._G
    alpha_n = model._alpha_n
    values = model.cbGetSolution(x)
    removed = frozenset(i for i in G if values[i] > 0.5)
    residual = G.subgraph(set(G) - removed)
    oversized = [c for c in nx.connected_components(residual) if len(c) > alpha_n]
    if not oversized:
        if len(removed) < len(model._best):
            model._best = removed
        return

    # Reject the current integer candidate before considering an early exit.
    for component in oversized:
        for cut in _component_cuts(residual, component, alpha_n, model._rng):
            model.cbLazy(gp.quicksum(x[i] for i in cut) >= 1)
            if time.perf_counter() >= model._deadline:
                model.terminate()
                return

    gcc = max(oversized, key=len)
    local_ub = min(len(model._best & gcc), len(gcc) - alpha_n)
    local_lb = _degree_lower_bound(residual.subgraph(gcc), alpha_n, local_ub)
    if local_lb > model._lb:
        # Proposition 4.4 transfers the component bound to the original graph.
        model.cbLazy(model._objective >= local_lb)
        model._lb = local_lb
        model._objective_cut_bound = max(model._objective_cut_bound, local_lb)
    if model._lb == len(model._best):
        model.terminate()
        return

    repaired = set(removed)
    for component in oversized:
        if time.perf_counter() >= model._deadline:
            model.terminate()
            return
        _, _, local_removed = New(residual.subgraph(component).copy(), alpha_n)
        repaired.update(local_removed)
    if time.perf_counter() >= model._deadline:
        model.terminate()
        return
    candidate = _restore_candidate(G, alpha_n, repaired, model._leaf_neighbors)
    if len(candidate) < len(model._best):
        if not _is_feasible(G, candidate, alpha_n):
            raise RuntimeError("The repair heuristic produced an infeasible removal set.")
        model._best = candidate
        model.cbSetSolution(model._variable_list, [int(i in candidate) for i in G])


def _is_feasible(G, removed, alpha_n):
    residual = G.subgraph(set(G) - removed)
    return all(len(c) <= alpha_n for c in nx.connected_components(residual))


def main_BC(
    G,
    alpha_n=None,
    *,
    time_limit=500.0,
    threads=32,
    seed=1,
    solver_seed=0,
    use_auxiliary_constraints=True,
    return_status=False,
):
    """Run ENDB&C on a connected, undirected NetworkX graph.

    The default output remains (removed_nodes, LB, UB, elapsed_seconds).
    Pass return_status=True to append Optimal, TimeLimit, or Feasible.
    Elapsed time includes initialization and the master solve. Gurobi
    receives the remaining budget after initialization.
    """
    started = time.perf_counter()
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite.")
    deadline = started + time_limit
    if G.is_directed() or G.is_multigraph() or nx.number_of_selfloops(G):
        raise ValueError("G must be a simple undirected graph.")
    n = len(G)
    if alpha_n is None:
        alpha_n = max(n // 100, 2)
    if not isinstance(alpha_n, Integral) or alpha_n < 1:
        raise ValueError("alpha_n must be a positive integer.")
    alpha_n = int(alpha_n)

    labels = list(G)
    integer_labels = set(labels) == set(range(n))

    def result(removed, lb, status):
        indices = sorted(removed)
        if integer_labels:
            nodes = np.asarray([labels[i] for i in indices], dtype=int)
        else:
            nodes = np.empty(len(indices), dtype=object)
            nodes[:] = [labels[i] for i in indices]
        output = (nodes, lb, len(indices), time.perf_counter() - started)
        return (*output, status) if return_status else output

    components = list(nx.connected_components(G))
    if all(len(component) <= alpha_n for component in components):
        return result((), 0, "Optimal")
    if len(components) != 1:
        raise ValueError("For nontrivial instances, supply a connected graph.")

    G = nx.relabel_nodes(G, dict(zip(labels, range(n))), copy=True)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    # For a connected graph on two nodes, excluding both leaves would remove
    # all feasible dismantling sets when alpha_n is 1.
    leaf_neighbors = {
        i: next(iter(G[i])) for i in G if G.degree[i] == 1
    } if use_auxiliary_constraints and n > 2 else {}
    _, _, ndsg_removed = New(G.copy(), alpha_n)
    best = _restore_candidate(G, alpha_n, ndsg_removed, leaf_neighbors)
    if not _is_feasible(G, best, alpha_n):
        raise RuntimeError("NDSG and node restoration did not produce a feasible warm start.")
    ub = len(best)
    lb = max(1, _degree_lower_bound(G, alpha_n, ub))
    if lb == ub:
        return result(best, lb, "Optimal")
    if time.perf_counter() >= deadline:
        return result(best, lb, "TimeLimit")

    with gp.Model("ENDB&C") as model:
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        objective = gp.quicksum(x[i] for i in G)
        model.setObjective(objective, GRB.MINIMIZE)
        model.addConstr(objective >= lb, name="initial_lb")
        model.addConstr(objective <= ub, name="initial_ub")
        if use_auxiliary_constraints:
            for leaf in leaf_neighbors:
                model.addConstr(x[leaf] == 0, name=f"leaf_{leaf}")
            for i in G:
                model.addConstr(
                    x[i] <= gp.quicksum(1 - x[j] for j in G[i]),
                    name=f"retained_neighbor_{i}",
                )

        for cut in _component_cuts(G, G.nodes, alpha_n, rng):
            model.addConstr(gp.quicksum(x[i] for i in cut) >= 1)
            if time.perf_counter() >= deadline:
                return result(best, lb, "TimeLimit")
        model.update()
        for i in G:
            x[i].BranchPriority = G.degree[i]
            x[i].Start = int(i in best)

        model._G = G
        model._alpha_n = alpha_n
        model._x_vars = x
        model._variable_list = [x[i] for i in G]
        model._objective = objective
        model._best = best
        model._lb = lb
        model._objective_cut_bound = lb
        model._leaf_neighbors = leaf_neighbors
        model._deadline = deadline
        model._rng = rng

        model.Params.Threads = threads
        model.Params.Seed = solver_seed
        model.Params.LazyConstraints = 1
        model.Params.PreCrush = 1
        model.Params.MIPGap = 0.0
        model.Params.Heuristics = 0.5
        model.Params.MIPFocus = 3
        model.Params.Cuts = -1
        model.Params.Presolve = 2
        model.Params.Aggregate = 1
        model.Params.BranchDir = 1
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return result(best, lb, "TimeLimit")
        model.Params.TimeLimit = remaining
        model.optimize(my_callback)

        best = model._best
        lb = model._lb
        final_solver_bound = _integer_solver_bound(model.ObjBound)
        if final_solver_bound is not None:
            lb = max(lb, final_solver_bound)
        if lb > len(best):
            raise RuntimeError("The reported lower bound exceeds the feasible incumbent.")
        if lb == len(best):
            status = "Optimal"
        elif model.Status == GRB.TIME_LIMIT or time.perf_counter() >= deadline:
            status = "TimeLimit"
        else:
            status = "Feasible"
        return result(best, lb, status)

if __name__ == '__main__':

    import pickle

    file_path = r'data/Euroroad.pkl'

    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    
    # G=nx.erdos_renyi_graph(20,0.2)
    # G = nx.barabasi_albert_graph(100, 2)
    # G = nx.grid_2d_graph(10, 10)
    
    
    # G = nx.convert_node_labels_to_integers(G)
    
    # G=nx.random_regular_graph(3,10)
    
    L = list(nx.connected_components(G))
    max_L = max(L, key=lambda x: len(x))
    
    G=G.subgraph(max_L)
    
    G = nx.convert_node_labels_to_integers(G)
    
    n = len(G.nodes)
    
    print(n,len(G.edges))
    alpha_n =max([int(0.01*n),2])
    # alpha_n=10
    # alpha_n=int(n*0.75)
    
    # SIMPLR_LB(G,alpha_n)

    print(main_BC(G, alpha_n=alpha_n)[0],len(main_BC(G, alpha_n=alpha_n)[0]))    
