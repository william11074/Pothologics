import pandas as pd
import numpy as np
import uuid
import datetime

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
if __name__ == "__main__":
    pothole_data = generate_pothole_data(500)
    pothole_data.to_csv('pothole_dataset.csv', index=False)
    print("Generated dataset with shape:", pothole_data.shape)

    # Also generate a detections-style CSV matching server schema (without accuracy)
    # Columns: detection_id,device_id,cls,start_timestamp,end_timestamp,duration_s,lat,lon,confidence_peak,image_url
    device_id = 'simulator-1'
    events = []
    now = datetime.datetime.utcnow()

    # Use first N points to simulate events
    for i, row in pothole_data.head(300).iterrows():
        start = now + datetime.timedelta(seconds=int(np.random.randint(0, 600)))
        duration = float(np.random.randint(2, 11))
        end = start + datetime.timedelta(seconds=duration)
        conf = float(np.random.uniform(0.6, 0.95))
        events.append({
            'detection_id': str(uuid.uuid4()),
            'device_id': device_id,
            'cls': 'Pothole',
            'start_timestamp': start.replace(tzinfo=datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
            'end_timestamp': end.replace(tzinfo=datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
            'duration_s': f"{duration:.3f}",
            'lat': float(row['latitude']),
            'lon': float(row['longitude']),
            'confidence_peak': conf,
            'image_url': ''
        })

    det_df = pd.DataFrame(events, columns=['detection_id','device_id','cls','start_timestamp','end_timestamp','duration_s','lat','lon','confidence_peak','image_url'])
    det_df.to_csv('detections_simulated.csv', index=False)
    print("Generated detections_simulated.csv with", len(det_df), "rows")
