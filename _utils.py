import json

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

class ParamSolution:

	def __init__(self, C, Q, previous_wire_len):

		self.C = C
		self.Q = Q
		self.curr_wirelen = previous_wire_len

	def __str__(self):

		return f"C: {self.C}, Q: {self.Q}, CW: {self.curr_wirelen}\n"

def get_wire_delay(params: SchemParams, wirelen, C_in):

	return (wirelen ** 2) * params.C_w * params.R_w / 2 + params.R_w * wirelen * (C_in + params.C_w * wirelen)

def get_capac_with_wire(params: SchemParams, CQ_sol: ParamSolution):

	return CQ_sol.C + params.C_w * CQ_sol.curr_wirelen

def get_delay_with_wire(params: SchemParams, CQ_sol: ParamSolution):

	return CQ_sol.Q - get_wire_delay(params, CQ_sol.curr_wirelen, get_capac_with_wire(params, CQ_sol))

def get_buff_delay(params: SchemParams, C_in):

	return params.R_b * C_in + params.D_b

def get_wire_len(edge_to_parent):

	num_segments = len(edge_to_parent)
	wire_length = 0

	wire_length += abs(edge_to_parent[0][0] - edge_to_parent[1][0])
	wire_length += abs(edge_to_parent[0][1] - edge_to_parent[1][1])

	if num_segments == 3:
		wire_length += abs(edge_to_parent[1][0] - edge_to_parent[2][0])
		wire_length += abs(edge_to_parent[1][1] - edge_to_parent[2][1])

	return wire_length