import json
import argparse
from collections import deque

class SteinerNode:

	def __init__(self, json_node, parent_node, path_to_parent):

		print(json_node)
		self.json = json_node
		self.parent = parent_node
		self.path_to_parent = path_to_parent
		self.childs = []

		if not parent_node == None:
			self.parent.childs.append(self)

class SchemParams:

	def __init__(self, tech_file, driver_node):

		json_data = None
		with open(tech_file, "r") as file:
			json_data = json.load(file)

		self.R_w = json_data["technology"]["unit_wire_resistance"]
		self.C_w = json_data["technology"]["unit_wire_capacitance"]

		self.R_b = json_data["module"][0]["input"][0]["R"]
		self.C_b = json_data["module"][0]["input"][0]["C"]
		self.D_b = json_data["module"][0]["input"][0]["intrinsic_delay"]

def fill_tree(parent, vertices, edges):

	if parent.json["type"] == "t":
		return

	curr_id = parent.json["id"]
	# find all childs
	edges_to_childs = [edge for edge in edges if edge["vertices"][0] == curr_id]
	print(edges_to_childs)
	print("\n")

	for edge_to_child in edges_to_childs:

		curr_child_id = edge_to_child["vertices"][1]
		# slow?
		child_json = next(vertice for vertice in vertices if vertice["id"] == curr_child_id)
		child = SteinerNode(child_json, parent, edge_to_child["segments"])

		fill_tree(child, vertices, edges)

def get_wire_len(path_to_parent):

	num_segments = len(path_to_parent)
	wire_length = 0

	wire_length += abs(path_to_parent[0][0] - path_to_parent[1][0])
	wire_length += abs(path_to_parent[0][1] - path_to_parent[1][1])

	if num_segments == 3:
		wire_length += abs(path_to_parent[1][0] - path_to_parent[2][0])
		wire_length += abs(path_to_parent[1][1] - path_to_parent[2][1])

	return wire_length

class ParamSolution:

	def __init__(self, C, Q, previous_wire_len):

		self.C = C
		self.Q = Q
		self.curr_wirelen = previous_wire_len

def get_wire_delay(params, wirelen, C_in):

	return (wirelen ** 2) * params.C_w * params.R_w / 2 + params.R_w * wirelen * C_in

def get_buff_delay(params, C_in):

	return params.R_b * C_in + params.D_b

def calc_branch(node, params, curr_solutions: [ParamSolution]): # Q - RAT

	wire_len = get_wire_len(node.path_to_parent)

	for i in range(wire_len): # or + 1 for s == t?

		cs_best_to_buf = curr_solutions[0]
		for cs in curr_solutions:

			# set wire
			cs.curr_wirelen += 1
			cs.Q = cs.Q - get_wire_delay(params, wirelen, cs.C)
			cs.C = cs.C + params.C_w * cs.curr_wirelen # Is ok?

			# <= ? Or >=?
			if cs.Q - get_buff_delay(params, cs.C) <= cs_best_to_buf.Q - get_buff_delay(params, cs_best_to_buf.C)
				cs_best_to_buf = cs

		cs_buf = ParamSolution(params.C_b, cs_best_to_buf.Q - get_buff_delay(params, cs_best_to_buf.C, 0))
		curr_solutions.append(cs_buf)

	return curr_solutions

def merge_two_childs(left_solutions, right_solutions, params):

	solutions = []

	for curr_solution in [*left_solutions, *right_solutions]:

		curr_Q = curr_solution.Q
		C_more_than_Q = [solution.C for solution in [*left_solutions, *right_solutions] if solution.Q >= curr_Q]

		if len(more_than_Q) == 0:
			continue

		curr_C = curr_solution.C + min(C_more_than_Q)
		solutions.append(ParamSolution(curr_C, curr_Q, ??0))

	return solutions

def merge_childs(child_solutions):

	for i in range(len(child_solutions) - 1):
		child_solutions[0] = merge_two_childs(child_solutions[0], child_solutions[i])

	return child_solutions


def bottom_up_order(node: SteinerNode):

	if node.json["type"] == "t":
		return [ParamSolution(node.json["capacitance"], node.json["rat"], 0)]

	solutions = []

	for child in node.childs:
		solutions.append(calc_branch(bottom_up_order(child)))

	merge_solutions()

	return solutions

class SteinerTree:

	def __init__(self, input_file): 

		json_data = None
		with open(input_file, "r") as file:
			json_data = json.load(file)

		vertices = json_data["node"]
		edges = json_data["edge"]

		# sort by the vertices from which the edges emerge
		edges.sort(key = lambda edge: edge["vertices"][0])

		# set root - node with type b
		root_json_node = next(vertice for vertice in vertices if vertice["type"] == "b")

		self.root = SteinerNode(root_json_node, None, None)

		fill_tree(self.root, vertices, edges)

		self.print()

	def print(self):

		print("STEINER TREE:")
		q = deque()

		q.append(self.root)

		while len(q):

			node = q.popleft()

			print("parent: ", node.json["id"], " and childs:", [child.json["id"] for child in node.childs])

			for child in node.childs:
				q.append(child)

	def get_vanGinneken_path():
		
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "parse files")
	parser.add_argument("technology_filename")
	parser.add_argument("test_filename")

	args = parser.parse_args()

	tree = SteinerTree(args.test_filename)

	bottom_up_order(tree.root)