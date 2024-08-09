import xml.etree.ElementTree as ET
import argparse

def main(file_path):
    # Load the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract departure times
    depart_times = [float(trip.get('depart')) for trip in root.findall('trip')]

    # Calculate total vehicles from trips and time span
    total_vehicles = len(depart_times)
    time_span = max(depart_times) - min(depart_times) if depart_times else 0

    # Calculate trips per second
    trips_per_second = total_vehicles / time_span if time_span > 0 else 0

    # Dictionary to hold route information
    flow_data = {}

    # Iterate through each vehicle in the route file
    trip_count = 0 
    for trip in root.findall('trip'):
        trip_count += 1
        from_edge = trip.get('from')
        to_edge = trip.get('to')
        
        if (from_edge, to_edge) not in flow_data:
            flow_data[(from_edge, to_edge)] = 0
        flow_data[(from_edge, to_edge)] += 1

    assert total_vehicles == trip_count, print(f"total vehicles, {total_vehicles} \
                                               and trip count {trip_count} do not match")
    print(f"Total Vehicles: {total_vehicles}")
    print(f"Time Span: {time_span:.2f} seconds")
    print(f"Trips per Second: {trips_per_second:.3f}")
    print(f"Total Unique Routes: {len(flow_data.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate trips per second from a SUMO route file.')
    parser.add_argument('file_path', type=str, help='Path to the SUMO route XML file.')
    args = parser.parse_args()
    main(args.file_path)