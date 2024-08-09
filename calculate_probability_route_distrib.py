import xml.etree.ElementTree as ET
import os 
import numpy as np
import pprint as pprint 
import argparse
import matplotlib.pyplot as plt 
from pprint import pprint

def get_flow_data(path_to_flow_file, simulation_time):
    tree = ET.parse(os.path.abspath(path_to_flow_file))
    root = tree.getroot()

    flows = {}
    total_no_flows = 0
    unique_probs = []
    for flow in root.findall("flow"):
        total_no_flows +=1 
        from_edge, to_edge = (flow.get('from'), flow.get('to'))
        probability = flow.get('probability')
        if probability not in unique_probs:
            unique_probs.append(probability)
        no_vehicles_in_route = probability * simulation_time
        flows[(from_edge, to_edge)] = no_vehicles_in_route
    
    return flows, total_no_flows, unique_probs

def main(path_to_flow_file, simulation_time):
    flows, total_no_flows, unique_probs  = get_flow_data(path_to_flow_file, simulation_time)
    print("total_no_flows:", total_no_flows)
    print("unique_probs:", unique_probs)
    pprint(flows)

if __name__ == "__main__":

    argumentParser = argparse.ArgumentParser(
        "Outputs the different probabilites, as well as the unique routes found in flow file")
    argumentParser.add_argument('path_to_flow_file')
    argumentParser.add_argument('simulation_time')

    parsed_args = argumentParser.parse_args()
    path_to_flow_file = parsed_args.path_to_flow_file
    simulation_time = int(parsed_args.simulation_time)

    print(type(path_to_flow_file))
    main(path_to_flow_file, simulation_time)



