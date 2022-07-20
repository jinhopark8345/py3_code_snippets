import networkx as nx
import networkx.algorithms as algo

import matplotlib.pyplot as plt


def draw(G):
    nx.draw(G, with_labels=True)
    plt.show()


def demo1():
    # H = nx.path_graph(5)
    # G.add_nodes_from([4,3, 7,8,9])
    # G.add_edges_from([[4,3], [7,8], [9,4], [4,8]])

    # Z = nx.union(H, G)
    # g1 = nx.junction_tree(G)
    # Z2 = nx.disjoint_union(H, G)
    # print(Z.nodes, Z.edges)
    # print(Z2.nodes, Z2.edges)

    # print(g1.nodes, g1.edges)
    # tmp = nx.algorithms.approximation.treewidth_min_fill_in(G)

    # G = nx.Graph()
    G = nx.DiGraph()
    G.add_edges_from(
        [
            [2, 5],
            [1, 3],
            [1, 4],
            [3, 9],
            [2, 6],
            [3, 7],
            [3, 8],
            [1, 2],
        ]
    )

    # (Pdb++) list(nx.all_pairs_shortest_path_length(G))
    # [(1, {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}), (2, {2: 0, 5: 1, 6: 1}), (3, {3: 0, 8: 1, 9: 1, 7: 1}), (4, {4: 0}), (5, {5: 0}), (6, {6: 0}), (7, {7: 0}), (8, {8: 0}), (9, {9: 0})]


    # (Pdb++) list(nx.all_pairs_shortest_path_length(G))
    # [(2, {2: 0, 5: 1, 6: 1}), (5, {5: 0}), (1, {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}), (3, {3: 0, 8: 1, 9: 1, 7: 1}), (4, {4: 0}), (9, {9: 0}), (6, {6: 0}), (7, {7: 0}), (8, {8: 0})]
    G2 = nx.Graph()
    G2.add_edges_from(
        [[1, 2], [2, 5], [5, 9]]
    )

    G3 = nx.Graph()
    G3.add_edges_from(
        [[1, 2], [1, 3], [1, 4], [2, 5], [2, 6], [3, 7], [3, 8], [3, 9]]
    )

    # find leaf nodes, need to make G = nx.DiGraph()
    # leaf_nodes = [
    #     x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1
    # ]

    breakpoint()

    print(f"{leaf_nodes = }")
    print(f"{nx.is_tree(G)}")
    print(f"{nx.is_forest(G)}")

    undirectedG = G.to_undirected()
    print(f"{list(nx.bridges(undirectedG)) = }")


def demo2():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            [1, 2],
            [1, 3], [3, 4], [4, 5],
            [2, 6], [6, 7], [7, 8],
        ]
    )

    H = nx.DiGraph()
    H.add_edges_from(
        [[1, 2], [1, 3], [1, 4], [2, 5], [2, 6], [3, 7], [3, 8], [3, 9]]
    )

    print(f'{list(nx.strongly_connected_components(G)) = }')
    breakpoint()



    S = nx.intersection(G, H)
    draw(S)

def demo_subgraph():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            [1, 2],
            [1, 3], [3, 4], [4, 5],
            [2, 6], [6, 7], [7, 8],
        ]
    )

    K = nx.induced_subgraph(G, [3,4,5])
    list(K.nodes)
    draw(K)

# demo_subgraph()

demo1()
