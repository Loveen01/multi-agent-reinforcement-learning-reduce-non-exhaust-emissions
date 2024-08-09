import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# Load and parse the XML file
file_path = "path_to_your_file.xml"
tree = ET.parse(file_path)
root = tree.getroot()

# Extract the start and end points of the routes
routes = []
for flow in root.findall('flow'):
    from_route = flow.get('from')
    to_route = flow.get('to')
    routes.append((from_route, to_route))

# Create a DataFrame from the routes
routes_df = pd.DataFrame(routes, columns=['From', 'To'])

# Generate a grid diagram showing the start and end points
def create_grid_diagram(routes_df):
    ids = ["A3", "B3", "C3", "D3",
           "A2", "B2", "C2", "D2",
           "A1", "B1", "C1", "D1",
           "A0", "B0", "C0", "D0"]
    
    # Create a mapping from IDs to coordinates
    id_to_coords = {id_: (3 - int(id_[1]), ord(id_[0]) - ord('A')) for id_ in ids}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['A', 'B', 'C', 'D'])
    ax.set_yticks(range(4))
    ax.set_yticklabels(['3', '2', '1', '0'])
    
    for _, row in routes_df.iterrows():
        start_coords = id_to_coords[row['From']]
        end_coords = id_to_coords[row['To']]
        ax.arrow(start_coords[1], start_coords[0], end_coords[1] - start_coords[1], end_coords[0] - start_coords[0],
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

# Display the routes data
print(routes_df.head())

# Create the grid diagram
create_grid_diagram(routes_df)
