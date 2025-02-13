import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.coord_transform import wgs84_to_gcj02
from shapely.wkt import loads
from shapely.geometry import Polygon, Point
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def draw_heat_map(data, i):
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap='RdYlGn_r', interpolation='nearest')
    ax.axis('off')
    cbar = plt.colorbar(cax)
    plt.subplots_adjust(bottom=0.15, right=0.85, top=0.95, left=0.15)
    plt.savefig("figures/"+str(i)+"-heatmap.png")

def heat_map_reshape(pred_heat_map, grid_size):
    # Define target grid dimensions
    target_rows = grid_size[0]
    target_cols = grid_size[1]
    new_shape = (target_rows, target_cols)
    B = np.zeros(new_shape)

    # Calculate merge factors
    row_factor = pred_heat_map.shape[0] / target_rows
    col_factor = pred_heat_map.shape[1] / target_cols

    # Merge cells
    for i in range(target_rows):
        for j in range(target_cols):
            start_row = int(np.floor(i * row_factor))
            end_row = int(np.floor((i + 1) * row_factor))
            start_col = int(np.floor(j * col_factor))
            end_col = int(np.floor((j + 1) * col_factor))
            B[i, j] = pred_heat_map[start_row:end_row, start_col:end_col].sum()
    
    return np.rint(B).astype(int)

def calculate_flows(data):
    # Calculate cumulative flows from inflow/outflow data
    D, T, H, W, _ = data.shape
    flows = np.zeros((D, T, H, W))
    
    for d in range(D):
        flows[d, 0] = data[d, 0, ..., 0]
        for t in range(1, T):
            flows[d, t] = flows[d, t-1] + data[d, t, ..., 0] - data[d, t, ..., 1]
    
    return flows

class CombinedDataset(Dataset):
    def __init__(self, data, pred_data):
        self.data = data
        self.pred_data = pred_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        pred_data_item = self.pred_data[idx]
        return data_item, pred_data_item

def process_data(data, pred=False):
    # Reshape data into daily timesteps
    D = 304  # Number of days
    T = 24   # Hours per day
    try:
        data = data.reshape(D, T, data.shape[1], data.shape[2])
    except ValueError as e:
        raise ValueError(f"Failed to reshape data: {e}")

    # Extract data between 10:00 and 22:00
    start_hour = 10
    end_hour = 22
    data = data[:, (start_hour - 1):(end_hour-1), :, :]
    
    # Sample every 3rd timestep
    data = data[:, ::3, :, :]
    
    # Remove negative values
    data[data < 0] = 0

    if pred:
        # Split into validation and test (50:50)
        val_num = int(D * 0.5)
        val_data = data[:val_num]
        test_data = data[val_num:]
        return val_data, test_data
    else:
        # Split into train, validation and test (60:20:20)
        train_num = int(D * 0.6)
        val_num = int(D * 0.2)
        train_data = data[:train_num]
        val_data = data[train_num:train_num + val_num]
        test_data = data[train_num + val_num:]
        return train_data, val_data, test_data

def get_dataloader(datapath, start_day, end_day, nb_ts_per_day, batch_size, grid_size):
    # Load data files
    start_time_str = start_day.strftime("%Y%m%d")
    end_time_str = end_day.strftime("%Y%m%d")
    data_path = datapath + "frames_{}_{}_{}.npy".format(
        start_time_str, end_time_str, nb_ts_per_day
    )
    
    flows = np.load(data_path)
    inflows = flows[...,0]
   
    with open('data/model_predictions.pkl', 'rb') as f:
        pred_flows = pickle.load(f)

    # Process prediction data
    pre_plus = pred_flows[0]
    pre_plus = pre_plus[:, :, :, ::3]
    pre_plus = np.transpose(pre_plus, (0, 3, 1, 2))
    pre_plus = np.tile(pre_plus, (2, 1, 1, 1))[:62]
    
    # Process actual data
    train_data, val_data, test_data = process_data(flows)
    
    # Initialize neighbor grid
    neighbor_grid = [[[] for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    max_len = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            neighbor_grid[i][j] = [i, j]
            max_len = max(max_len, len(neighbor_grid[i][j]))

    # Create datasets
    train_dataset = CombinedDataset(train_data, train_data)
    val_dataset = CombinedDataset(val_data, val_data)
    test_dataset = CombinedDataset(test_data, pre_plus)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader, neighbor_grid

def generate_masked_heat_map(frames):
    mask = get_mask(frames)
    filtered = frames * mask
    return filtered

def get_mask(frames):
    # Define study area polygon (coordinates anonymized)
    study_area = loads('POLYGON ((116.4 39.8, 116.4 39.8, 116.4 39.8, 116.4 39.8))')
    x, y = study_area.exterior.xy
    study_area_transformed = []
    for i in range(len(x)):
        study_area_transformed.append(wgs84_to_gcj02(x[i], y[i]))
    study_area_polygon = Polygon(study_area_transformed)
    _, H, W = frames.shape
    mask = generate_region_mask(H, W, study_area_polygon)
    return mask

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_neighbors_within_distance(grid_size, center, neighbor_num):
    neighbors = []
    half_size = int((neighbor_num-1)/2)

    for i in range(center[0] - half_size, center[0] + half_size + 1):
        for j in range(center[1] - half_size, center[1] + half_size + 1):
            if 0 <= i < grid_size[0] and 0 <= j < grid_size[1]:
                neighbors.append([i, j])
            else:
                neighbors.append([-1, -1])

    while len(neighbors) < neighbor_num * neighbor_num:
        neighbors.append([-1, -1])

    return neighbors

def generate_region_mask(H, W, polygon):
    mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            lat, lng = revert_transform(j, i, -53, 26, 39.86, 116.49)
            pt = Point(lng, lat)
            if polygon.contains(pt):
                mask[i, j] = 1
    return mask

def revert_transform(matrix_x, matrix_y, x_min, y_max, center_lat, center_lng):
    ori_x = matrix_x + x_min
    ori_y = y_max - matrix_y
    return decode_lat(center_lat, ori_y), decode_lng(center_lng, ori_x)

def decode_lat(center_lat, y):
    return (1e4 * center_lat + y) / 1e4

def decode_lng(center_lng, x):
    return (1e4 * center_lng + x) / 1e4
