import json
import argparse
from collections import deque
import numpy as np
from copy import copy

from _utils import *

class SteinerNode:

	def __init__(self, json_node, parent_node, edge_to_parent, params):

#		print(json_node)
		self.json = json_node
		self.parent = parent_node
		self.edge_to_parent = edge_to_parent
		self.childs = []

		# this array contain of 0 and 1 for each position for buffer 
		# in edge from terminals to driver
		self.edge_buf_options = []
		
		# this contain position in childs_options corespond with edge_buf_options
		self.edge_buf_to_childs_num = []

		# this array contains elements that contains options from each child
		self.childs_options = []

		# params from tech file
		self.tech_params = params

		if not parent_node == None:
			self.parent.childs.append(self)

	def __str__(self):

		return str(self.json)

	def calc_branch(self, curr_solutions: [ParamSolution]): # Q - RAT

		wire_len = get_wire_len(self.edge_to_parent)

#		print("wire_len:", wire_len, "curr solutions in calc_branch:")
#		print(*curr_solutions)

		init_size = len(curr_solutions)
		edge_solutions = [[]] * init_size
		edge_indexes = [i for i in range(len(curr_solutions))]

		# need index - from input curr_solutions - for edge_indexes and index for dynamically changed edge_solutions
		# [start; finish); start from terminals
		for i in range(wire_len):

			cs_best_to_buf = curr_solutions[0]
			cs_index = 0

			for j in range(len(curr_solutions)):

				# set wire
				cs = curr_solutions[j]
				edge_solutions[j].append(0)

				cs.curr_wirelen += 1
				curr_Q = get_delay_with_wire(self.tech_params, cs)
				curr_C = get_capac_with_wire(self.tech_params, cs)

				curr_best_Q = get_delay_with_wire(self.tech_params, cs_best_to_buf)
				curr_best_C = get_capac_with_wire(self.tech_params, cs_best_to_buf)

				bufferized_curr_Q = curr_Q - get_buff_delay(self.tech_params, curr_C)
				bufferized_best_Q = curr_best_Q - get_buff_delay(self.tech_params, curr_best_C)

				# maximal Q is best variant
				if bufferized_curr_Q >= bufferized_best_Q:
					cs_best_to_buf = cs
					cs_index = j

			new_edge_solution = copy(edge_solutions[cs_index])
			new_edge_solution[-1] = 1

			edge_solutions.append(new_edge_solution)
			edge_indexes.append(edge_indexes[cs_index])

			curr_best_Q = get_delay_with_wire(self.tech_params, cs_best_to_buf)
			curr_best_C = get_capac_with_wire(self.tech_params, cs_best_to_buf) # With wire? - Yes

			# Start wirelen is 0 there
			cs_buf = ParamSolution(self.tech_params.C_b, curr_best_Q - get_buff_delay(self.tech_params, curr_best_C), 0)
			curr_solutions.append(cs_buf)

			for cs in curr_solutions:

				curr_Q = get_delay_with_wire(self.tech_params, cs)
				curr_C = get_capac_with_wire(self.tech_params, cs)

			# Don't update C and Q on wires because don't null curr_wirelen

		self.edge_buf_options = edge_solutions
		self.edge_buf_to_childs_num = edge_indexes

		return curr_solutions

	def merge_childs(self, childs_CQ_params_arr):

		num_of_childs = len(self.childs)

		solutions = []

		solution_index = 0
		
		for curr_target_child_index in range(len(childs_CQ_params_arr)):
			target_CQ_params_arr = childs_CQ_params_arr[curr_target_child_index]

			for curr_CQ_param_index in range(len(target_CQ_params_arr)):
				target_CQ_param = target_CQ_params_arr[curr_CQ_param_index]

				have_solution = True

				curr_C = get_capac_with_wire(self.tech_params, target_CQ_param)
				curr_Q = get_delay_with_wire(self.tech_params, target_CQ_param)

				solution_ways = [int] * num_of_childs
				solution_ways[curr_target_child_index] = curr_CQ_param_index
				
				max_prev_wirelen = target_CQ_param.curr_wirelen

				for curr_other_child_index in range(len(childs_CQ_params_arr)):
					other_CQ_params_arr = childs_CQ_params_arr[curr_other_child_index]

					# Very slow? need compare id's?
					if other_CQ_params_arr == target_CQ_params_arr:
						continue

					params_C_more_than_Q = [CQ_param for CQ_param in other_CQ_params_arr if get_delay_with_wire(self.tech_params, CQ_param) >= curr_Q]

					if len(params_C_more_than_Q) == 0:
						have_solution = False
						break # twice-up-break

					# Choose best variant by C 
					min_C = min(get_capac_with_wire(self.tech_params, C_more_than_Q) for C_more_than_Q in params_C_more_than_Q)
					min_index = [i for i in range(len(other_CQ_params_arr)) if get_capac_with_wire(self.tech_params, other_CQ_params_arr[i]) == min_C]
					min_index = min_index[0]

					curr_C += get_capac_with_wire(self.tech_params, other_CQ_params_arr[min_index])
					# maybe min?
					max_prev_wirelen = max(max_prev_wirelen, other_CQ_params_arr[min_index].curr_wirelen)

					solution_ways[curr_other_child_index] = min_index

				if not have_solution:
					continue

				solutions.append(ParamSolution(curr_C, curr_Q, max_prev_wirelen))

				self.childs_options.append(solution_ways) # must be solution_index

				solution_index += 1

		if len(solutions) != solution_index:
			print("\nPANIC solution size don't equal solution index\n")
		return solutions


	def get_buffer_coords(self, index):

		edges = self.edge_to_parent

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

	def get_curr_edge_buffers(self, solution_id):

		buffers = []

		wire_len = get_wire_len(self.edge_to_parent)

		# Check start and end
		for i in range(wire_len):
			if self.edge_buf_options[solution_id][i] == 1:
				buffers.append(self.get_buffer_coords(i))

		return buffers

