import json
import argparse
from collections import deque
import numpy as np
from copy import copy

class SteinerNode:

	def __init__(self, json_node, parent_node, path_to_parent):

		print(json_node)
		self.json = json_node
		self.parent = parent_node
		self.path_to_parent = path_to_parent
		self.childs = []

		# maybe one class?
		self.options_to_parent = []
		self.num_option_from_child = []

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

	return (wirelen ** 2) * params.C_w * params.R_w / 2 + params.R_w * wirelen * (C_in + params.C_w * wirelen)

def get_buff_delay(params: SchemParams, C_in):

	return params.R_b * C_in + params.D_b

def calc_branch(node: SteinerNode, params: SchemParams, curr_solutions: [ParamSolution]): # Q - RAT

	wire_len = get_wire_len(node.path_to_parent)

	print("wire_len:", wire_len)
	print("CURR SOLS in calc_branch:")
	print(*curr_solutions)
#	print(node)

	init_size = len(curr_solutions)
	edge_solutions = [[]] * init_size
	edge_indexes = [i for i in range(len(curr_solutions))]

	print("Start ES:", edge_solutions)

	# need index - from input curr_solutions - for edge_indexes and index for dynamically changed edge_solutions
	# [start; finish); start from terminals
	for i in range(wire_len): # or + 1 for s == t?

		cs_best_to_buf = curr_solutions[0]
		cs_index = 0

		for j in range(len(curr_solutions)):

			# set wire
			cs = curr_solutions[j]
			edge_solutions[j].append(0)

			cs.curr_wirelen += 1
			curr_Q = cs.Q - get_wire_delay(params, cs.curr_wirelen, cs.C)
			curr_C = cs.C + params.C_w * cs.curr_wirelen

			curr_best_Q = cs_best_to_buf.Q - get_wire_delay(params, cs_best_to_buf.curr_wirelen, cs_best_to_buf.C)
			curr_best_C = cs_best_to_buf.C + params.C_w * cs_best_to_buf.curr_wirelen

			bufferized_curr_Q = curr_Q - get_buff_delay(params, curr_C)
			bufferized_best_Q = curr_best_Q - get_buff_delay(params, curr_best_C)

			# minimal Q is best variant
			if bufferized_curr_Q <= bufferized_best_Q:
				cs_best_to_buf = cs
				cs_index = j

		print("cs_index:", cs_index)

		if cs_index < 0:
			print("OH NO CS INDEX < 0!!!")

		new_edge_solution = copy(edge_solutions[cs_index])
		new_edge_solution[-1] = 1

		edge_solutions.append(new_edge_solution)
		edge_indexes.append(edge_indexes[cs_index]) # TODO: It is not correct !

		curr_best_Q = cs_best_to_buf.Q - get_wire_delay(params, cs_best_to_buf.curr_wirelen, cs_best_to_buf.C)
		curr_best_C = cs_best_to_buf.C + params.C_w * cs_best_to_buf.curr_wirelen

		cs_buf = ParamSolution(params.C_b, curr_best_Q - get_buff_delay(params, curr_best_C), 0)
		curr_solutions.append(cs_buf)

		# Don't update C and Q on wires because don't null curr_wirelen but in merge?..

	print("edge solutions:", len(edge_solutions))
	print("indexes:", len(edge_indexes))

	for i in range(len(edge_solutions)):
		print("i-th sol:", i, "child:", edge_indexes[i], ":", end = " ")
		for j in range(len(edge_solutions[i])):
			if edge_solutions[i][j] == 1:
				print(j, end = " ")
		print("")

	print("CURR SOLS after calculation in calc_branch:")
	print(*curr_solutions)

	node.options_to_parent = edge_solutions
	node.num_option_from_child = edge_indexes
#	print(node.num_option_from_child)

	return curr_solutions

def merge_childs(childs_CQ_params_arr, params: SchemParams, node: SteinerNode):

	num_of_childs = len(node.childs)

	solutions = []

	print("\n")
	print("num_of_childs: ", num_of_childs)

	solution_index = 0
	
	for curr_target_child_index in range(len(childs_CQ_params_arr)):
		target_CQ_params_arr = childs_CQ_params_arr[curr_target_child_index]

		for curr_CQ_param_index in range(len(target_CQ_params_arr)):
			target_CQ_param = target_CQ_params_arr[curr_CQ_param_index]

			have_solution = True

			curr_C = target_CQ_param.C
			curr_Q = target_CQ_param.Q - get_wire_delay(params, target_CQ_param.curr_wirelen, target_CQ_param.C)

			solution_ways = [int] * num_of_childs
			solution_ways[curr_target_child_index] = curr_CQ_param_index
			
			max_prev_wirelen = target_CQ_param.curr_wirelen

			for curr_other_child_index in range(len(childs_CQ_params_arr)):
				other_CQ_params_arr = childs_CQ_params_arr[curr_other_child_index]

				# Very slow? need compare id's?
				if other_CQ_params_arr == target_CQ_params_arr:
					continue

				params_C_more_than_Q = [CQ_param for CQ_param in other_CQ_params_arr if CQ_param.Q - get_wire_delay(params, CQ_param.curr_wirelen, CQ_param.C) >= curr_Q]

				if len(params_C_more_than_Q) == 0:
					have_solution = False
					break # twice-up-break

				# Choose best variant by C 
				min_C = min(C_more_than_Q.C + params.C_w * C_more_than_Q.curr_wirelen for C_more_than_Q in params_C_more_than_Q)
				min_index = [i for i in range(len(other_CQ_params_arr)) if (other_CQ_params_arr[i].C + params.C_w * other_CQ_params_arr[i].curr_wirelen) == min_C]
				print("min_index:", min_index)
				min_index = min_index[0]


				curr_C += other_CQ_params_arr[min_index].C + params.C_w * other_CQ_params_arr[min_index].curr_wirelen
				# maybe min?
				max_prev_wirelen = max(max_prev_wirelen, other_CQ_params_arr[min_index].curr_wirelen)

				solution_ways[curr_other_child_index] = min_index

			if not have_solution:
				continue

			solutions.append(ParamSolution(curr_C, curr_Q, max_prev_wirelen))

#			print("in node", node.json["id"], "on index", solution_index, ":", solution_ways)
			node.options_from_merge.append(solution_ways) # must be solution_index

			solution_index += 1

#	print(node.options_from_merge)

	if len(solutions) != solution_index:
		print("\nPANIC solution size don't equal solution index\n")
	return solutions

def bottom_up_order(node: SteinerNode, tech_params: SchemParams):

	if node.json["type"] == "t":
		return [ParamSolution(node.json["capacitance"], node.json["rat"], 0)]

	solutions = []

	for child in node.childs:
		solutions.append(calc_branch(child, tech_params, bottom_up_order(child, tech_params)))

	solved = merge_childs(solutions, tech_params, node)
	# print(*solved)
	print("from node", node.json["id"], ", solutions:", len(solved))

	return solved

def get_buffer_coords(node: SteinerNode, index):

	edges = node.path_to_parent

#	print("edges:", edges)

	num_segments = len(edges)

	# first index - edge num, second index - x {0} or y {1}

	# can be optimized
	if num_segments == 2:
		# starting point - #1, end point - #0 - because we start from terminals
		if edges[0][0] == edges[1][0]:
			len_sign = int(np.sign(edges[0][1] - edges[1][1]))
			return [edges[1][0], edges[1][1] + len_sign * index]
		else:
			len_sign = int(np.sign(edges[0][0] - edges[1][0]))
			return [edges[1][0] + len_sign * index, edges[1][1]]

	if num_segments == 3:
		# starting point - #2, middle - #1, end - #0

		# maybe <?
		if index <= abs(edges[1][0] - edges[2][0]) + abs(edges[1][1] - edges[2][1]):

			if edges[2][0] == edges[1][0]:
				len_sign = int(np.sign(edges[1][1] - edges[2][1]))
				return [edges[2][0], edges[2][1] + len_sign * index]
			else:
				len_sign = int(np.sign(edges[1][0] - edges[2][0]))
				return [edges[2][0] + len_sign * index, edges[2][1]]
		else:
			index -= abs(edges[1][0] - edges[2][0]) + abs(edges[1][1] - edges[2][1])
			if edges[1][0] == edges[0][0]:
				len_sign = int(np.sign(edges[0][1] - edges[1][1]))
				return [edges[1][0], edges[1][1] + len_sign * index]
			else:
				len_sign = int(np.sign(edges[0][0] - edges[1][0]))
				return [edges[1][0] + len_sign * index, edges[1][1]]

def get_curr_edge_buffers(node: SteinerNode, solution_id):

	buffers = []

	wire_len = get_wire_len(node.path_to_parent)

	print("node:", node.json)
#	solution_id -= 10
	print("solution_id:", solution_id)
	# Check start and end
	for i in range(wire_len):
		if node.options_to_parent[solution_id][i] == 1:
			buffers.append(get_buffer_coords(node, i))
	print("buffers:", buffers)

	return buffers

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

	def print(self):

		print("STEINER TREE:")
		q = deque()

		q.append(self.root)

		while len(q):

			node = q.popleft()

			print("parent: ", node.json["id"], " and childs:", [child.json["id"] for child in node.childs])

			for child in node.childs:
				q.append(child)

def add_started_nodes(node_array, node):

	node_array.append(node.json)

	print(node.json)

	for child in node.childs:
		add_started_nodes(node_array, child)

class EdgeBufferAndSplit:

	def __init__(self): 

		self.buffers = []
		self.vertices = []

def get_curr_edge_split(node: SteinerNode, curr_vert_index, curr_edge_index, edges_array, buffers_places):

	new_vertices = [node.path_to_parent[0]]

	# there too first index - edge num, second index - x {0} or y {1}

	segments = node.path_to_parent
	num_segments = len(segments)

	curr_starting_vert = node.json["id"]
	curr_end_vert = curr_vert_index

	wire_len = get_wire_len(node.path_to_parent)

	if num_segments == 2:

		# starting point - #1, end point - #0 - because we start from terminals
		curr_len = 0

		curr_x = segments[1][0]
		curr_y = segments[1][1]

		for i in range(wire_len):

			if i == wire_len - 1:
				curr_end_vert = node.json["id"]

			if buffers_places[i] == 1:

				if segments[1][0] == segments[0][0]:
					len_sign = int(np.sign(segments[0][1] - segments[1][1]))
					new_edge = {"id": curr_edge_index, 
								"vertices": [ curr_starting_vert, curr_end_vert], 
								"segments": [ [curr_x, curr_y], [curr_x, curr_y + len_sign * curr_len]]}
					curr_y += len_sign * curr_len
				else:
					len_sign = int(np.sign(segments[0][0] - segments[1][0]))
					new_edge = {"id": curr_edge_index, 
								"vertices": [ curr_starting_vert, curr_end_vert], 
								"segments": [ [curr_x, curr_y], [curr_x + len_sign * curr_len, curr_y]]}
					curr_x += len_sign * curr_len

				edges_array.append(new_edge)
				
				curr_starting_vert = curr_end_vert
				curr_end_vert += 1

				curr_edge_index += 1

				curr_len = 0

			curr_len += 1

		len_sign = int(np.sign(segments[0][0] - segments[1][0]))
		new_edge = {"id": curr_edge_index, 
					"vertices": [ curr_starting_vert, node.json["id"]], 
					"segments": [ [curr_x, curr_y], [segments[0][0], segments[0][1]]]}

		edges_array.append(new_edge)
		curr_edge_index += 1

	else:
		# starting point - #2, middle - #1, end - #0
		if num_segments != 3:
			print("PANIC, num_segments is not 3, but must be!")

		curr_len = 0

		first_segment_len = abs(segments[2][0] - segments[1][0]) + abs(segments[2][1] - segments[1][1])

		curr_x = segments[2][0]
		curr_y = segments[2][1]

		for i in range(wire_len):

			if i == wire_len - 1:
				curr_end_vert = node.json["id"]

			# TODO: Error? In case of 3 segments? In second part?
			curr_start = abs(curr_x - segments[2][0]) + abs(curr_y - segments[2][1])
			curr_end = curr_start + curr_len

			if buffers_places[i] == 1:

				if curr_end <= first_segment_len:

					if segments[2][0] == segments[1][0]:
						len_sign = int(np.sign(segments[1][1] - segments[2][1]))
						new_edge = {"id": curr_edge_index, 
									"vertices": [ curr_starting_vert, curr_end_vert], 
									"segments": [ [curr_x, curr_y], [curr_x, curr_y + len_sign * curr_len]]}
						curr_y += len_sign * curr_len
					else:
						len_sign = int(np.sign(segments[1][0] - segments[2][0]))
						new_edge = {"id": curr_edge_index, 
									"vertices": [ curr_starting_vert, curr_end_vert], 
									"segments": [ [curr_x, curr_y], [curr_x + len_sign * curr_len, curr_y]]}
						curr_x += len_sign * curr_len
				elif curr_start < first_segment_len and curr_end > first_segment_len:

					middle_point = None
					if segments[2][0] == segments[1][0]:
						len_sign = int(np.sign(segments[1][1] - segments[2][1]))
						middle_point = [segments[2][0], segments[2][1] + len_sign * first_segment_len]
					else:
						len_sign = int(np.sign(segments[1][0] - segments[2][0]))
						middle_point = [segments[2][0] + len_sign * first_segment_len, segments[2][1]]

					second_part_len = curr_end - first_segment_len

					if segments[1][0] == segments[0][0]:
						len_sign = int(np.sign(segments[0][1] - segments[1][1]))
						new_edge = {"id": curr_edge_index,
									"vertices": [ curr_starting_vert, curr_end_vert], 
									"segments": [ [curr_x, curr_y], middle_point, [segments[1][0], segments[1][1] + len_sign * second_part_len]]}
						curr_y += len_sign * curr_len
					else:
						len_sign = int(np.sign(segments[0][0] - segments[1][0]))
						new_edge = {"id": curr_edge_index, 
									"vertices": [ curr_starting_vert, curr_end_vert],
									"segments": [ [curr_x, curr_y], middle_point, [segments[1][0] + len_sign * second_part_len, segments[1][1]]]}
						curr_x += len_sign * curr_len
				else:

					if segments[1][0] == segments[0][0]:
						len_sign = int(np.sign(segments[0][1] - segments[1][1]))
						new_edge = {"id": curr_edge_index, 
									"vertices": [ curr_starting_vert, curr_end_vert], 
									"segments": [ [curr_x, curr_y], [curr_x, curr_y + len_sign * curr_len]]}
						curr_y += len_sign * curr_len
					else:
						len_sign = int(np.sign(segments[0][0] - segments[1][0]))
						new_edge = {"id": curr_edge_index, 
									"vertices": [ curr_starting_vert, curr_end_vert], 
									"segments": [ [curr_x, curr_y], [curr_x + len_sign * curr_len, curr_y]]}
						curr_x += len_sign * curr_len

				edges_array.append(new_edge)
				
				curr_starting_vert = curr_end_vert
				curr_end_vert += 1

				curr_edge_index += 1

				curr_len = 0

			curr_len += 1
		# TODO: Add last edge like in case with 2 segments

	return curr_edge_index

def get_nodes_and_edges(nodes_array, edges_array, curr_vert_index, curr_edge_index, node, edge_solution_id):

	curr_buffers = get_curr_edge_buffers(node, edge_solution_id)
	curr_edge_index = get_curr_edge_split(node, curr_vert_index, curr_edge_index, edges_array, node.options_to_parent[edge_solution_id])

	for buffer in curr_buffers:
		buf = {"id": curr_vert_index, "x": buffer[0], "y": buffer[1], "type": "b", "name": "buf1x"}
		nodes_array.append(buf)
		curr_vert_index += 1

	if node.json["type"] == "t":
		return curr_vert_index, curr_edge_index
	
	steiner_point_sol_id = node.num_option_from_child[edge_solution_id]
	indexes_in_childs_edges = node.options_from_merge[steiner_point_sol_id]
	for i in range(len(node.childs)):

		# choose solution the child
		child_sol_id = indexes_in_childs_edges[i]

		# buffers ordered from start of edge to end
		curr_vert_index, curr_edge_index = get_nodes_and_edges(nodes_array, edges_array, 
						curr_vert_index, curr_edge_index, node.childs[i], child_sol_id)

	return curr_vert_index, curr_edge_index

def dump_tree_to_json(tree, filename, solution_id):
	
	node = [] # "node" in json
	edge = [] # "edge" in json

	add_started_nodes(node, tree.root)

	start_index = len(node)

	print("BEFORE BUFFERS PRINTING")
	get_nodes_and_edges(node, edge, start_index, 0, tree.root.childs[0], tree.root.options_from_merge[solution_id][0])

	with open("calculated_" + filename, 'w') as file:
	    json.dump({"node" : node, "edge" : edge}, file, sort_keys = False, indent = 4)

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
	max_RAT = params[0].Q - get_wire_delay(tech_params, params[0].curr_wirelen, params[0].C)
	# Need to delete options with same RAT
	i = 0
	for param in params:
		if param.Q - get_wire_delay(tech_params, param.curr_wirelen, param.C + tech_params.C_w * param.curr_wirelen) >= max_RAT:
			max_RAT = param.Q - get_wire_delay(tech_params, param.curr_wirelen, param.C + tech_params.C_w * param.curr_wirelen)
			max_index = i
		i += 1

	print("solutions:")
	print(*params)

	dump_tree_to_json(tree, args.test_filename, max_index)
