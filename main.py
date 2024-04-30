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

		self.options_to_parent = []
		self.options_from_merge = []

		if not parent_node == None:
			self.parent.childs.append(self)

	def __str__(self):

		return str(self.json)

class SchemParams:

	def __init__(self, tech_file):

		json_data = None
		with open(tech_file, "r") as file:
			json_data = json.load(file)

		self.R_w = json_data["technology"]["unit_wire_resistance"]
		self.C_w = json_data["technology"]["unit_wire_capacitance"]

		self.R_b = json_data["module"][0]["input"][0]["R"]
		self.C_b = json_data["module"][0]["input"][0]["C"]
		self.D_b = json_data["module"][0]["input"][0]["intrinsic_delay"]

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

	def __str__(self):

		return f"C: {self.C}, Q: {self.Q}, CW: {self.curr_wirelen}\n"

def get_wire_delay(params: SchemParams, wirelen, C_in):

	return (wirelen ** 2) * params.C_w * params.R_w / 2 + params.R_w * wirelen * C_in

def get_buff_delay(params: SchemParams, C_in):

	return params.R_b * C_in + params.D_b

def calc_branch(node: SteinerNode, params: SchemParams, curr_solutions: [ParamSolution]): # Q - RAT

	wire_len = get_wire_len(node.path_to_parent)

	print("wire_len:", wire_len)
	print(*curr_solutions)
	print(node)

	init_size = len(curr_solutions)
	edge_solutions = [[]] * init_size

	print("init size: ", init_size)
	print(edge_solutions)

	# [start; finish); start from terminals
	for i in range(wire_len): # or + 1 for s == t?

		cs_best_to_buf = curr_solutions[0]
		cs_index = 0

		for j in range(len(curr_solutions)):

			cs = curr_solutions[j]
			# set wire
			cs.curr_wirelen += 1
			cs.Q = cs.Q - get_wire_delay(params, wire_len, cs.C)
			cs.C = cs.C + params.C_w * cs.curr_wirelen # Is ok?

			edge_solutions[j].append(0)

			# <= ? Or >=?
			if cs.Q - get_buff_delay(params, cs.C) <= cs_best_to_buf.Q - get_buff_delay(params, cs_best_to_buf.C):
				cs_best_to_buf = cs
				cs_index = j

		if cs_index < 0:
			print("OH NO CS INDEX < 0!!!")

		edge_solutions.append(edge_solutions[cs_index])
		edge_solutions[-1][-1] = 1 # Буффер есть

		cs_buf = ParamSolution(params.C_b, cs_best_to_buf.Q - get_buff_delay(params, cs_best_to_buf.C), 0)
		curr_solutions.append(cs_buf)

	print("edge solutions:", len(edge_solutions))
#	print("edge solutions:", edge_solutions)

	node.options_to_parent = edge_solutions

	return curr_solutions

def merge_childs(childs_CQ_params_arr, params: SchemParams, num_of_childs):

	solutions = []

	solution_index = 0
	curr_target_child_index = 0
	for child_CQ_params_arr in childs_CQ_params_arr:

		curr_CQ_param_index = 0
		for child_CQ_param in child_CQ_params_arr:

			curr_C = curr_solution.C
			curr_Q = child_CQ_param.Q

			solution_ways = [None] * len(num_of_childs)
			solution_ways[curr_target_child_index] = curr_CQ_param_index
			
			curr_other_child_index = 0
			for other_CQ_params_arr in childs_CQ_params_arr:
				# Very slow? need compare id's?
				if other_CQ_params_arr == child_CQ_params_arr:
					continue

				C_more_than_Q = [CQ_param.C for CQ_param in other_CQ_params_arr if CQ_param.Q >= curr_Q]

				if len(more_than_Q) == 0:
					break # twice up - break

				min_index = C_more_than_Q.index(min(C_more_than_Q))
				curr_C += C_more_than_Q[min_index]
			
				solution_ways[curr_other_child_index] = min_index	
				curr_other_child_index += 1
			
			solutions.append(ParamSolution(curr_C, curr_Q, 0)) # !!! может быть здесь верно выбирать больший или меньший
			
			node.options_from_merge.append(solution_ways) # must be solution_index

			solution_index += 1
			curr_CQ_param_index += 1
		curr_target_child_index += 1

	if len(solutions) != solution_index:
		print("\nPANIC solution size don't equal solution index\n")
	return solutions

def bottom_up_order(node: SteinerNode, tech_params: SchemParams):

	if node.json["type"] == "t":
		return [ParamSolution(node.json["capacitance"], node.json["rat"], 0)]

	solutions = []

	for child in node.childs:
		solutions.append(calc_branch(child, tech_params, bottom_up_order(child, tech_params)))

	solved = merge_childs(solutions, tech_params, len(node.childs))
	print(solved)

	return solved

def get_buffer_coords(node: SteinerNode, index):

	edges = node.path_to_parent

	num_segments = len(edges)

	# can be optimized
	if num_segments == 2:
		if edges[0].x == edges[1].x:
			return [edges[0].x, edges[0].y + index]
		else:
			return [edges[0].x + index, edges[0].y]

	if num_segments == 3:

		# maybe <?
		if index <= abs(edges[0].x - edges[1].x + edges[0].y - edges[1].y):

			if edges[0].x == edges[1].x:
				return [edges[0].x, edges[0].y + index]
			else:
				return [edges[0].x + index, edges[0].y]
		else:
			index -= abs(edges[0].x - edges[1].x + edges[0].y - edges[1].y)
			if edges[1].x == edges[2].x:
				return [edges[1].x, edges[1].y + index]
			else:
				return [edges[1].x + index, edges[1].y]

def get_curr_edge_buffers(node: SteinerNode, solution_id):

	buffers = []

	wire_len = get_wire_len(node.path_to_parent)

	# Check start and end
	for i in range(wire_len):
		if node.options_to_parent[solution_id][i] == 1:
			buffers.append(get_buffer_coords(node, i))

	return buffers

def get_curr_edge_split(node: SteinerNode, buffers_coords):

	buffers_places = node.options_to_parent[solution_id]
	new_vertices = [node.path_to_parent[0]]

	segments = node.path_to_parent
	num_segments = len(segments)

	wire_len = get_wire_len(node.path_to_parent)

	if num_segments == 2:
		
		curr_len = 0
		if segments[0].x == segments[1].x:
			for i in range(wire_len):
				# TODO ???
				if buffers_places[i] == 1:
					new_vertices.append([segments[0].x, segments[0].y + curr_len])
					curr_len = 0
				curr_len += 1
		else:
			for i in range(wire_len):
				# TODO ???
				if buffers_places[i] == 1:
					new_vertices.append([segments[0].x + curr_len, segments[0].y])
					curr_len = 0
				curr_len += 1
		# Or it will be there??
		new_vertices.append([segments[1].x, segments[1].y])

	else:
		curr_len = 0
		first_segment_len = abs(segments[0].x - segments[1].x + segments[0].y - segments[1].y)
		if segments[0].x == segments[1].x:
			for i in range(first_segment_len):
				# TODO ???
				if buffers_places[i] == 1:
					new_vertices.append([segments[0].x, segments[0].y + curr_len])
					curr_len = 0
				curr_len += 1
		else:
			for i in range(first_segment_len):
				# TODO ???
				if buffers_places[i] == 1:
					new_vertices.append([segments[0].x + curr_len, segments[0].y])
					curr_len = 0
				curr_len += 1
		# Or it will be there??
		new_vertices.append([segments[1].x, segments[1].y])

		curr_len = 0
		second_segment_len = abs(segments[2].x - segments[1].x + segments[2].y - segments[1].y)
		if segments[0].x == segments[1].x:
			for i in range(first_segment_len, second_segment_len):
				# TODO ???
				if buffers_places[i] == 1:
					new_vertices.append([segments[0].x, segments[0].y + curr_len])
					curr_len = 0
				curr_len += 1
		else:
			for i in range(first_segment_len, second_segment_len):
				# TODO ???
				if buffers_places[i] == 1:
					new_vertices.append([segments[0].x + curr_len, segments[0].y])
					curr_len = 0
				curr_len += 1
		# Or it will be there??
		new_vertices.append([segments[1].x, segments[1].y])

	return new_vertices

class EdgeBufferAndSplit:

	def __init__(self): 

		self.buffers = []
		self.vertices = []

def get_solution_buffers(node: SteinerNode, solution_id, buffers_and_edges):

	for i in range(len(node.childs)):
		subtree_sol_id = node.options_from_merge[i]

		# buffers ordered from start of edge to end
		get_solution_buffers(node.childs[i], subtree_sol_id, buffers_and_edges)

	curr_buffers = get_curr_edge_buffers(node, solution_id)
	buffers_and_edges.buffers.append(curr_buffers)
	buffers_and_edges.vertices.append(get_curr_edge_split(node, curr_buffers))

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

#		self.print()

	def print(self):

		print("STEINER TREE:")
		q = deque()

		q.append(self.root)

		while len(q):

			node = q.popleft()

			print("parent: ", node.json["id"], " and childs:", [child.json["id"] for child in node.childs])

			for child in node.childs:
				q.append(child)

def print_buffers_from_solution(tree: SteinerTree, solution_id):

	root = tree.root

	buffers_and_edges = EdgeBufferAndSplit()
	# Root have only one child
	get_solution_buffers(root.child[0], buffers_and_edges)

	return buffers_and_edges.buffers, buffers_and_edges.vertices

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "parse files")
	parser.add_argument("technology_filename")
	parser.add_argument("test_filename")

	args = parser.parse_args()

	tree = SteinerTree(args.test_filename)

	# Get array of params
	tech_params = SchemParams(args.technology_filename)
	params = bottom_up_order(tree.root, tech_params)

	# Get index of max RAT
	max_index = 0
	max_RAT = params[0].Q
	# Need to delete options with same RAT
	i = 0
	for param in params:
		if param.Q >= max_RAT:
			max_RAT = param.Q
			max_index = i
		i += 1

	buffers, edges = print_buffers_from_solution(tree, max_index)