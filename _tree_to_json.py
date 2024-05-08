
from _steiner_node import *
from _utils import *

from pathlib import Path as path

def add_started_nodes(node_array, node):

	node_array.append(node.json)

	for child in node.childs:
		add_started_nodes(node_array, child)

def get_curr_edge_split(node: SteinerNode, curr_vert_index, curr_edge_index, edges_array, buffers_places):

	new_vertices = [node.edge_to_parent[0]]

	# there too first index - edge num, second index - x {0} or y {1}

	segments = node.edge_to_parent
	num_segments = len(segments)

	curr_starting_vert = node.json["id"]
	curr_end_vert = curr_vert_index

	wire_len = get_wire_len(node.edge_to_parent)

	if num_segments == 2:

		# starting point - #1, end point - #0 - because we start from terminals
		curr_len = 0

		curr_x = segments[1][0]
		curr_y = segments[1][1]

		for i in range(wire_len):

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
					"vertices": [ curr_starting_vert, node.parent.json["id"]],
					"segments": [ [curr_x, curr_y], [segments[0][0], segments[0][1]]]}

		edges_array.append(new_edge)
		curr_edge_index += 1

	else:
		# starting point - #2, middle - #1, end - #0
		if num_segments != 3:
			print("PANIC, num_segments is not 3, but must be! It is", num_segments)
			print(segments)

		curr_len = 0

		first_segment_len = abs(segments[2][0] - segments[1][0]) + abs(segments[2][1] - segments[1][1])

		curr_x = segments[2][0]
		curr_y = segments[2][1]

		for i in range(wire_len):

			# absolute len from start of edge
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

					middle_point = segments[1]
					second_part_len = curr_end - first_segment_len

					if segments[1][0] == segments[0][0]:
						len_sign = int(np.sign(segments[0][1] - segments[1][1]))
						new_edge = {"id": curr_edge_index,
									"vertices": [ curr_starting_vert, curr_end_vert], 
									"segments": [ [curr_x, curr_y], middle_point, [segments[1][0], segments[1][1] + len_sign * second_part_len]]}
						curr_y = segments[1][1] + len_sign * second_part_len
						curr_x = segments[1][0]
					else:
						len_sign = int(np.sign(segments[0][0] - segments[1][0]))
						new_edge = {"id": curr_edge_index, 
									"vertices": [ curr_starting_vert, curr_end_vert],
									"segments": [ [curr_x, curr_y], middle_point, [segments[1][0] + len_sign * second_part_len, segments[1][1]]]}
						curr_x = segments[1][0] + len_sign * second_part_len
						curr_y = segments[1][1]
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

		# abs from start
		curr_start = abs(curr_x - segments[2][0]) + abs(curr_y - segments[2][1])
		if curr_start > first_segment_len:

			len_sign = int(np.sign(segments[0][0] - segments[1][0]))
			new_edge = {"id": curr_edge_index, 
						"vertices": [ curr_starting_vert, node.parent.json["id"]],
						"segments": [ [curr_x, curr_y], [segments[0][0], segments[0][1]]]}

			edges_array.append(new_edge)
			curr_edge_index += 1
		else:

			middle_point = segments[1]
			end_point = segments[0]

			if segments[1][0] == segments[0][0]:
				len_sign = int(np.sign(segments[0][1] - segments[1][1]))
				new_edge = {"id": curr_edge_index,
							"vertices": [ curr_starting_vert, node.parent.json["id"]], 
							"segments": [ [curr_x, curr_y], middle_point, end_point]}
			else:
				len_sign = int(np.sign(segments[0][0] - segments[1][0]))
				new_edge = {"id": curr_edge_index, 
							"vertices": [ curr_starting_vert, node.parent.json["id"]],
							"segments": [ [curr_x, curr_y], middle_point, end_point]}

			edges_array.append(new_edge)
			curr_edge_index += 1

	return curr_edge_index

def get_nodes_and_edges(nodes_array, edges_array, curr_vert_index, curr_edge_index, node, edge_solution_id):

	curr_buffers = node.get_curr_edge_buffers(edge_solution_id)
	curr_edge_index = get_curr_edge_split(node, curr_vert_index, curr_edge_index, edges_array, node.edge_buf_options[edge_solution_id])

	for buffer in curr_buffers:
		buf = {"id": curr_vert_index, "x": buffer[0], "y": buffer[1], "type": "b", "name": "buf1x"}
		nodes_array.append(buf)
		curr_vert_index += 1

	if node.json["type"] == "t":
		return curr_vert_index, curr_edge_index
	
	steiner_point_sol_id = node.edge_buf_to_childs_num[edge_solution_id]
	indexes_in_childs_edges = node.childs_options[steiner_point_sol_id]
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

	get_nodes_and_edges(node, edge, start_index, 0, tree.root.childs[0], tree.root.childs_options[solution_id][0])

	# Must work on Windows
	out_filename = path(filename).name.rsplit(".json", 1)[0] + "_out.json"

	with open(out_filename, 'w') as file:
	    json.dump({"node" : node, "edge" : edge}, file, sort_keys = False, indent = 4)
