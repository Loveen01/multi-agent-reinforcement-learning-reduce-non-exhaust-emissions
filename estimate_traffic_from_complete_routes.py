import xml.etree.ElementTree as ET
import argparse

def main(file_path):
    # Load the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract departure times
    depart_times = [float(vehicle.get('depart')) for vehicle in root.findall('vehicle')]

    # Calculate total vehicles and time span
    total_vehicles = len(depart_times)
    time_span = max(depart_times) - min(depart_times) if depart_times else 0

    # Calculate trips per second
    trips_per_second = total_vehicles / time_span if time_span > 0 else 0

    # Dictionary to hold route information
    flow_data = {}

    # Iterate through each vehicle in the route file
    for vehicle in root.findall('vehicle'):
        route_edges = vehicle.find('route').get('edges').split()
        from_edge = route_edges[0]
        to_edge = route_edges[-1]
        
        if (from_edge, to_edge) not in flow_data:
            flow_data[(from_edge, to_edge)] = 0
        flow_data[(from_edge, to_edge)] += 1
    
    print(f"Total Vehicles: {total_vehicles}")
    print(f"Time Span: {time_span:.2f} seconds")
    print(f"Trips per Second: {trips_per_second:.3f}")
    print(f"Total Unique Routes: {len(flow_data.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate trips per second from a SUMO route file.')
    parser.add_argument('file_path', type=str, help='Path to the SUMO route XML file.')
    args = parser.parse_args()
    main(args.file_path)