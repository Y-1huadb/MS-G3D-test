import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

from graph import tools


# Part index:
# 0 head
# 1 torso
# 2 right_arm
# 3 left_arm
# 4 right_leg
# 5 left_leg

num_node = 6
self_link = [(i, i) for i in range(num_node)]
neighbor = [
    (0, 1),
    (1, 0),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (4, 5),
    (5, 4),
]


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
