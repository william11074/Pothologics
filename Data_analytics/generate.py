import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

def generate_pothole_data(num_points=500, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate randomly scattered points
    scattered_lat = np.random.uniform(38.70, 38.80, int(num_points * 0.6))
    scattered_lon = np.random.uniform(-122.50, -122.35, int(num_points * 0.6))
    
    # Generate clustered points
    cluster_centers = [
        (38.75, -122.42),
        (38.72, -122.38),
        (38.77, -122.45),
        (38.74, -122.48)
    ]
    
    clustered_points = []
    for center in cluster_centers:
        # Create a cluster with spread
        lat, lon = center
        cluster_size = int(num_points * 0.1)
        cluster_lat = np.random.normal(lat, 0.005, cluster_size)
        cluster_lon = np.random.normal(lon, 0.005, cluster_size)
        clustered_points.extend(zip(cluster_lat, cluster_lon))
    
    all_lat = np.concatenate([scattered_lat, [p[0] for p in clustered_points]])
    all_lon = np.concatenate([scattered_lon, [p[1] for p in clustered_points]])
    
    # Create csv file
    df = pd.DataFrame({
        'latitude': all_lat,
        'longitude': all_lon,
        'report_date': pd.date_range(start='2023-01-01', periods=len(all_lat)),
        'report_time': np.random.choice(pd.date_range("07:00", "21:00", freq="T").time, len(all_lat)),
    })
    
    return df.sample(frac=1).reset_index(drop=True)

# Generate and save the dataset
pothole_data = generate_pothole_data(500)
pothole_data.to_csv('pothole_dataset.csv', index=False)
print("Generated dataset with shape:", pothole_data.shape)