import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint
import datetime

def _resolve_default_csv(explicit_path: str | None = None) -> str:
    """Resolve which detections CSV to use automatically.
    Priority: explicit_path > detections_source.json > env DETECTIONS_SOURCE > whichever exists.
    """
    if explicit_path:
        return str(explicit_path)
    # Toggle file
    try:
        src_file = Path('detections_source.json')
        if src_file.exists():
            with src_file.open('r', encoding='utf-8') as f:
                data = json.load(f) or {}
                src = str(data.get('source') or '').lower()
                if src == 'real':
                    return 'detections.csv'
                if src == 'simulated':
                    return 'detections_simulated.csv'
    except Exception:
        pass
    # Env fallback
    src = str(os.getenv('DETECTIONS_SOURCE', 'simulated')).lower()
    candidate = 'detections.csv' if src == 'real' else 'detections_simulated.csv'
    if Path(candidate).exists():
        return candidate
    # Last resort
    if Path('detections.csv').exists():
        return 'detections.csv'
    return 'detections_simulated.csv'


class PotholeRepairOptimizer:
    # filepath may be None for auto-selection
    def __init__(self, filepath: str | None = None):
        csv_path = _resolve_default_csv(filepath)
        self.df = pd.read_csv(csv_path)
        self.cluster_routes = {}
        self.route_stats = {}
        self.cluster_polygons = {}
        
        # Help reroute column names
        rename_map = {}
        if 'lat' in self.df.columns and 'latitude' not in self.df.columns:
            rename_map['lat'] = 'latitude'
        if 'lon' in self.df.columns and 'longitude' not in self.df.columns:
            rename_map['lon'] = 'longitude'
        if rename_map:
            self.df = self.df.rename(columns=rename_map)
        if 'start_timestamp' in self.df.columns and 'report_date' not in self.df.columns:
            try:
                self.df['report_date'] = pd.to_datetime(self.df['start_timestamp'], errors='coerce').dt.date.astype(str)
            except Exception:
                self.df['report_date'] = self.df['start_timestamp']
        
    # pothole clustering algorithm using DBSCAN
    # eps_km is the maximum distance between two samples for one to be considered
    # min_samples is the minimum number of potholes per cluster needed
    def cluster_potholes(self, eps_km=0.3, min_samples=3):
        if not {'latitude','longitude'}.issubset(self.df.columns):
            raise ValueError("Input must have latitude/longitude columns (or lat/lon).")
        coords = self.df[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        kms_per_radian = 6371.0088
        epsilon = eps_km / kms_per_radian
        
        db = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            algorithm='ball_tree',
            metric='haversine'
        ).fit(coords_rad)
        
        self.labels = db.labels_
        
        for cluster_id in set(self.labels):
            if cluster_id == -1:
                continue
            idx = np.where(self.labels == cluster_id)[0]
            cluster_points = self.df.iloc[idx][['latitude', 'longitude']].values
            if len(cluster_points) >= 3:
                try:
                    poly = MultiPoint(cluster_points).convex_hull
                    self.cluster_polygons[cluster_id] = poly
                except:
                    pass
        
        return self.df
    
    # Calculate distance between two potholes in km (accounting for earth curvature)
    def haversine_distance(self, point1, point2):
        lat1, lon1 = np.radians(point1)
        lat2, lon2 = np.radians(point2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    
    # generate optimized repair routes for each cluster
    # speed_kmh is the average travel speed between pothole locations
    # repair_time_mins is the average time to repair reach pothole
    # input data is used to estimate the repair time per route
    def optimize_routes(self, speed_kmh=30, repair_time_mins=15):
        unique_clusters = [int(c) for c in np.unique(self.labels) if c != -1]
        for idx, cluster_label in enumerate(sorted(unique_clusters)):
            route_id = idx + 1

            idxs = np.where(self.labels == cluster_label)[0]
            cluster_points = self.df.iloc[idxs]
            coords = cluster_points[['latitude', 'longitude']].values

            dist_matrix = cdist(coords, coords, lambda u, v: self.haversine_distance(u, v))

            n = len(coords)
            
            # traveling salesman problem optimization
            # for 9 or less potholes, brute forcing O(n!) is favorable
            # for more potholes, greedy algorithm is implemented
            if n < 10:
                from itertools import permutations
                best_order = None
                best_len = float('inf')
                for perm in permutations(range(1, n)):
                    candidate = [0] + list(perm)
                    length = sum(dist_matrix[candidate[i]][candidate[i+1]] for i in range(n-1))
                    if length < best_len:
                        best_len = length
                        best_order = candidate
                route = best_order if best_order is not None else list(range(n))
            else:
                route = [0]
                unvisited = set(range(1, n))
                while unvisited:
                    last = route[-1]
                    next_point = min(unvisited, key=lambda x: dist_matrix[last][x])
                    route.append(next_point)
                    unvisited.remove(next_point)

            # Store ordered route
            self.cluster_routes[route_id] = cluster_points.iloc[route].copy()
            self.cluster_routes[route_id]['route_order'] = range(len(route))

            # Calculate route statistics
            total_distance = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            total_time = (total_distance / speed_kmh * 60) + (len(route) * repair_time_mins)

            self.route_stats[route_id] = {
                'num_potholes': len(route),
                'total_km': round(total_distance, 2),
                'total_time_mins': round(total_time),
                'speed_kmh': speed_kmh,
                'repair_time_mins': repair_time_mins
            }

        return self.cluster_routes, self.route_stats
    
    # put the work orders into csv file
    def generate_work_orders_csv(self, filename="work_orders.csv"):
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")

        # Load road colors from JSON file
        colors_map = {}
        try:
            colors_path = Path('route_colors.json')
            if colors_path.exists():
                with colors_path.open('r', encoding='utf-8') as f:
                    colors_map = {str(k): str(v) for k, v in json.load(f).items()}
        except Exception:
            colors_map = {}

        work_orders = []
        for route_id, route_df in self.cluster_routes.items():
            route_df = route_df.copy()
            route_df['route_id'] = route_id
            # Attach route color if available
            route_df['route_color'] = colors_map.get(str(route_id), None)

            stats = self.route_stats.get(route_id, {})
            route_df['total_km'] = stats.get('total_km')
            route_df['total_time_mins'] = stats.get('total_time_mins')
            route_df['repair_time_mins'] = stats.get('repair_time_mins')

            work_orders.append(route_df)

        work_orders = pd.concat(work_orders, ignore_index=True)
        work_orders = work_orders.sort_values(['route_id', 'route_order'])

        preferred_cols = [
            'route_id', 'route_order', 'route_color',
            'latitude', 'longitude', 'report_date',
            'total_km', 'total_time_mins', 'repair_time_mins'
        ]
        cols = [c for c in preferred_cols if c in work_orders.columns] + \
               [c for c in work_orders.columns if c not in preferred_cols]
        work_orders = work_orders[cols]

        work_orders.to_csv(filename, index=False)
        return filename

    # route summary csv generation, for summary statistics only
    def generate_route_summary_csv(self, filename="route_summary.csv"):
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")

        rows = []
        for route_id, route_df in self.cluster_routes.items():
            stats = self.route_stats.get(route_id, {})

            r = route_df.sort_values('route_order')
            start_lat, start_lon = r.iloc[0][['latitude', 'longitude']]
            end_lat, end_lon = r.iloc[-1][['latitude', 'longitude']]
            centroid_lat = r['latitude'].mean()
            centroid_lon = r['longitude'].mean()

            rows.append({
                'route_id': route_id,
                'num_potholes': stats.get('num_potholes', len(r)),
                'total_km': stats.get('total_km'),
                'total_time_mins': stats.get('total_time_mins'),
                'repair_time_mins': stats.get('repair_time_mins'),
                'start_lat': start_lat,
                'start_lon': start_lon,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(['route_id'])
        df.to_csv(filename, index=False)
        return filename

# generate each cluster and optimize repair routes
# should be run every time a new pothole is detected or a route is deleted
if __name__ == "__main__":
    optimizer = PotholeRepairOptimizer(None)
    
    # cluster potholes
    optimizer.cluster_potholes()
    num_clusters = len([c for c in set(getattr(optimizer, 'labels', [])) if c != -1])
    print(f"Found {num_clusters} clusters")
    
    # optimize repair routes
    routes, stats = optimizer.optimize_routes()
    print("\nRoute Statistics:")
    for cluster_id, stat in stats.items():
        print(f"Cluster {cluster_id}: {stat['num_potholes']} potholes, "
              f"{stat['total_km']}km, {stat['total_time_mins']}mins")
    
    # generate work orders and route summary csv files
    work_orders_csv = optimizer.generate_work_orders_csv('work_orders.csv')
    summary_csv = optimizer.generate_route_summary_csv('route_summary.csv')
    print(f"\nWork orders saved to {work_orders_csv}")
    print(f"Route summary saved to {summary_csv}")
