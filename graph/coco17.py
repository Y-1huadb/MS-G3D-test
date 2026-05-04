import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

from graph import tools


# COCO17 joint index:
# {0,  "nose"}
# {1,  "left_eye"}
# {2,  "right_eye"}
# {3,  "left_ear"}
# {4,  "right_ear"}
# {5,  "left_shoulder"}
# {6,  "right_shoulder"}
# {7,  "left_elbow"}
# {8,  "right_elbow"}
# {9,  "left_wrist"}
# {10, "right_wrist"}
# {11, "left_hip"}
# {12, "right_hip"}
# {13, "left_knee"}
# {14, "right_knee"}
# {15, "left_ankle"}
# {16, "right_ankle"}
#
# Note: OpenPose18 neck is intentionally omitted because COCO17 has no neck joint.

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [
    (1, 0), (2, 0),
    (3, 1), (4, 2),
    (5, 6),
    (7, 5), (9, 7),
    (8, 6), (10, 8),
    (5, 11), (6, 12),
    (11, 12),
    (13, 11), (15, 13),
    (14, 12), (16, 14),
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

