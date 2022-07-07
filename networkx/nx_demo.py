import networkx as nx

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

    G = nx.DiGraph()
    G.add_edges_from(
        [[1, 2], [1, 3], [1, 4], [2, 5], [2, 6], [3, 7], [3, 8], [3, 9]]
    )

    # find leaf nodes
    leaf_nodes = [
        x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1
    ]

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

demo_subgraph()
