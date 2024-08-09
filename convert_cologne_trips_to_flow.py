import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import matplotlib.pyplot as plt 
from pprint import pprint
import pandas as pd 

route_path = "/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions-clean/data/cologne8/random_routes.rou.xml"

# ---------------------- CHANGE VALUES -------------------

# Original time span in the route file
ORIGINAL_TIME_SPAN = 3600  # Example value, adjust based on actual route file time span
# Define the new total simulation time
NEW_TOTAL_TIME = 10000  # New desired total simulation time in seconds

net_path = "data/cologne8/cologne8.net.xml"

# output_flow_file_path = f"data/4x4grid_similar_to_resco_for_train/flow_file_tps_constant_for_{NEW_TOTAL_TIME}s_with_scaled_route_distrib.rou.xml"
output_flow_file_path = f"data/cologne8/flow_file_tps_constant_for_{NEW_TOTAL_TIME}s.rou.xml"

# ---------------------- BEGIN -------------------
# Load the route file
tree = ET.parse(os.path.abspath(route_path))
root = tree.getroot()

# Dictionary to hold flow information
flow_data_route_path = {}

trip_count_route_path = 0
# Iterate through each vehicle in the route file
for trip in root.findall('trip'):
    trip_count_route_path += 1
    from_edge = trip.get("from")
    to_edge = trip.get("to")
    
    if (from_edge, to_edge) not in flow_data_route_path:
        flow_data_route_path[(from_edge, to_edge)] = 0
    flow_data_route_path[(from_edge, to_edge)] += 1

unique_trip_count_route_path = len(flow_data_route_path.keys())

# Create the root element for the flow file
flows_root = ET.Element('routes')

new_required_vehicle_trip_count = 2046 * (NEW_TOTAL_TIME / ORIGINAL_TIME_SPAN)

print("----------------Original file info----------------------")
print("no of vehicle trips in original route file: ", trip_count_route_path)
print("unique routes in original route file: ", unique_trip_count_route_path)
print("\n")
print("----------------New file info----------------------")
print("no of vehicle trips required in total for new sim: ", new_required_vehicle_trip_count)

vehicles_per_unique_route = trip_count_route_path / unique_trip_count_route_path

# # this is like arrival rate - or no of trips per second
# for every unique root there will be a flow ... 
# Generate flows from the collected data
for (from_edge, to_edge), count in flow_data_route_path.items():
    flow_id = f"flow_{from_edge}_{to_edge}"
    # original_probability = count / original_time  # Calculate original probability

    p_for_each_unique_flow = vehicles_per_unique_route * count / NEW_TOTAL_TIME

    # Use the original probability for the new flow
    flow_element = ET.SubElement(flows_root, 'flow', {
        'id': flow_id,
        'from': from_edge,
        'to': to_edge,
        'begin': '0',
        'end': str(NEW_TOTAL_TIME),
        'probability': str(p_for_each_unique_flow),
        'departSpeed': 'max',
        'departPos': 'base',
        'departLane': 'best'
    })



def pretty_print_custom(element, level=0):
    indent = "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = f"\n{indent}  "
        if not element.tail or not element.tail.strip():
            element.tail = f"\n{indent}"
        for elem in element:
            pretty_print_custom(elem, level + 1)
        if not element.tail or not element.tail.strip():
            element.tail = f"\n{indent}"
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = f"\n{indent}"

pretty_print_custom(flows_root)

# Write the flow file to disk
tree = ET.ElementTree(flows_root)

with open(output_flow_file_path, 'wb') as file:
    tree.write(file, encoding='utf-8', xml_declaration=True)

print(f"Flow file with constant probabilities generated successfully: {output_flow_file_path}")