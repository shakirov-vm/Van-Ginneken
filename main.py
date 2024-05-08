import json
import argparse
from collections import deque
import numpy as np
from copy import copy

from _utils import *
from _tree_to_json import *
from _steiner_node import *

def fill_tree(parent: SteinerNode, vertices, edges):

	if parent.json["type"] == "t":
		return

	curr_id = parent.json["id"]
	# find all childs
	edges_to_childs = [edge for edge in edges if edge["vertices"][0] == curr_id]
#	print(edges_to_childs)
#	print("\n")

	for edge_to_child in edges_to_childs:

		curr_child_id = edge_to_child["vertices"][1]
		# slow?
		child_json = next(vertice for vertice in vertices if vertice["id"] == curr_child_id)
		child = SteinerNode(child_json, parent, edge_to_child["segments"], parent.tech_params)

		fill_tree(child, vertices, edges)

def bottom_up_order(node: SteinerNode):

	if node.json["type"] == "t":
		return [ParamSolution(node.json["capacitance"], node.json["rat"], 0)]

	solutions = []

	for child in node.childs:
		solutions.append(child.calc_branch(bottom_up_order(child)))

	solved = node.merge_childs(solutions)
	# print(*solved)
#	print("from node", node.json["id"], ", solutions:", len(solved))

	return solved

class SteinerTree:

	def __init__(self, input_file, tech_file):

		tech_params = SchemParams(tech_file)

		json_data = None
		with open(input_file, "r") as file:
			json_data = json.load(file)

		vertices = json_data["node"]
		edges = json_data["edge"]

		# sort by the vertices from which the edges emerge
		edges.sort(key = lambda edge: edge["vertices"][0])

		# set root - node with type b
		root_json_node = next(vertice for vertice in vertices if vertice["type"] == "b")

		self.root = SteinerNode(root_json_node, None, None, tech_params)

		fill_tree(self.root, vertices, edges)

	def print(self):

		print("STEINER TREE:")
		q = deque()

		q.append(self.root)

		while len(q):

			node = q.popleft()

			print("parent: ", node.json["id"], " and childs:", [child.json["id"] for child in node.childs])

			for child in node.childs:
				q.append(child)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "parse files")
	parser.add_argument("technology_filename")
	parser.add_argument("test_filename")

	args = parser.parse_args()

	tree = SteinerTree(args.test_filename, args.technology_filename)

	params = bottom_up_order(tree.root)

	# Get index of max RAT
	# There is final C and Q, don't need to recalculate
	max_index = 0
	max_RAT = params[0].Q
	# Need to delete options with same RAT
	i = 0
	for param in params:
		if param.Q >= max_RAT:
			max_RAT = param.Q
			max_index = i
		i += 1

	print("solutions:")
	print(*params)
	print("best solution index:", max_index)
	print("best solution:", params[max_index])

	print("Best delay: ", 800 - max_RAT)

	dump_tree_to_json(tree, args.test_filename, max_index)

# With intrinsic_delay = 0.4 - max in middle, all is ok