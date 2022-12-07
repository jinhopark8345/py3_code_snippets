import networkx as nx
import networkx.algorithms as algo
import matplotlib.pyplot as plt
import logging
from logging.config import fileConfig

fileConfig('logging.ini')
logger = logging.getLogger('dev')


def get_start_nodes(dg: nx.DiGraph) -> list[int]:
    assert isinstance(dg, nx.DiGraph)
    return [node for node, in_degree in dg.in_degree() if in_degree==0]


def depth_first_search():
    dg = nx.DiGraph()
    dg.add_edges_from([(1,2),(2,3),(2,5)])
    root_nids = get_start_nodes(dg)

    if dg.has_node(root_nids[0]):
        child_parent = nx.dfs_predecessors(dg, root_nids[0])
        logger.debug(f'{child_parent = }')
        parent_child = nx.dfs_successors(dg, root_nids[0])
        logger.debug(f'{parent_child = }')
        dfs_tree_graph = nx.dfs_tree(dg, root_nids[0])
        logger.debug(f'{dfs_tree_graph.nodes = }')

    # nx.draw(dg, with_labels=True)
    # plt.show()

def demo_try():
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


def add_edges_from_with_directed_graph():
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
    nx.draw(G, with_labels=True)
    plt.show()

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
    nx.draw(G, with_labels=True)
    plt.show()


def main():
    # demo_try()
    # depth_first_search()

    for _ in range(1000) :
        options = {
            'node_color': 'black',
            'node_size': 100,
            'width': 3,
        }
        G = nx.dodecahedral_graph()
        shells = [[2, 3, 4, 5, 6], [8, 1, 0, 19, 18, 17, 16, 15, 14, 7], [9, 10, 11, 12, 13]]
        nx.draw_shell(G, nlist=shells, **options)
        nx.draw(G)
        plt.savefig("path.png")

if __name__ == '__main__':
    main()
