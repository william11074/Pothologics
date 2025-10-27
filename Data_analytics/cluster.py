import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint
import datetime

class PotholeRepairOptimizer:
    def __init__(self, filepath="./detections_simulated.csv"):
        self.df = pd.read_csv(filepath)
        self.cluster_routes = {}
        self.route_stats = {}
        self.cluster_polygons = {}
        # Normalize columns if using detections_simulated.csv (lat/lon -> latitude/longitude)
        rename_map = {}
        if 'lat' in self.df.columns and 'latitude' not in self.df.columns:
            rename_map['lat'] = 'latitude'
        if 'lon' in self.df.columns and 'longitude' not in self.df.columns:
            rename_map['lon'] = 'longitude'
        if rename_map:
            self.df = self.df.rename(columns=rename_map)

        # Provide optional fields if present in detections CSV
        if 'start_timestamp' in self.df.columns and 'report_date' not in self.df.columns:
            try:
                self.df['report_date'] = pd.to_datetime(self.df['start_timestamp'], errors='coerce').dt.date.astype(str)
            except Exception:
                self.df['report_date'] = self.df['start_timestamp']
        
    def cluster_potholes(self, eps_km=0.2, min_samples=3):
        # cluster potholes using DBSCAN
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
        
        self.df['cluster'] = db.labels_
        
        # Create cluster polygons
        for cluster_id in set(db.labels_):
            if cluster_id == -1:
                continue
            cluster_points = self.df[self.df['cluster'] == cluster_id][['latitude', 'longitude']].values
            if len(cluster_points) >= 3:
                try:
                    poly = MultiPoint(cluster_points).convex_hull
                    self.cluster_polygons[cluster_id] = poly
                except:
                    pass
        
        return self.df
    
    def haversine_distance(self, point1, point2):
        """Calculate haversine distance between two points in km"""
        lat1, lon1 = np.radians(point1)
        lat2, lon2 = np.radians(point2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    
    def optimize_routes(self, speed_kmh=30, repair_time_mins=15):
        """Optimize repair routes and assign sequential route IDs starting at 1."""
        unique_clusters = [cid for cid in self.df['cluster'].unique() if cid != -1]
        for idx, cluster_label in enumerate(sorted(unique_clusters)):
            route_id = idx + 1

            cluster_points = self.df[self.df['cluster'] == cluster_label]
            coords = cluster_points[['latitude', 'longitude']].values

            # Create distance matrix
            dist_matrix = cdist(coords, coords, lambda u, v: self.haversine_distance(u, v))

            # Solve TSP with greedy approach
            route = [0]  # Start with first point
            unvisited = set(range(1, len(coords)))

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
    # Removed HTML visualization; this script now focuses on CSV outputs only.
    
    def generate_work_orders(self):
        """Generate work orders with sequential route_id and no crew assignment."""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")

        work_orders = []
        for route_id, route_df in self.cluster_routes.items():
            route_df = route_df.copy()
            route_df['route_id'] = route_id
            work_orders.append(route_df)

        return pd.concat(work_orders).sort_values(['route_id', 'route_order'])
    
    def generate_work_orders_csv(self, filename="work_orders.csv"):
        """Write work orders to CSV (sequential route IDs, no crew)."""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")

        work_orders = []
        for route_id, route_df in self.cluster_routes.items():
            route_df = route_df.copy()
            route_df['route_id'] = route_id

            stats = self.route_stats.get(route_id, {})
            route_df['total_km'] = stats.get('total_km')
            route_df['total_time_mins'] = stats.get('total_time_mins')
            route_df['repair_time_mins'] = stats.get('repair_time_mins')

            work_orders.append(route_df)

        work_orders = pd.concat(work_orders, ignore_index=True)
        work_orders = work_orders.sort_values(['route_id', 'route_order'])

        preferred_cols = [
            'route_id', 'route_order',
            'latitude', 'longitude', 'report_date',
            'total_km', 'total_time_mins', 'repair_time_mins'
        ]
        cols = [c for c in preferred_cols if c in work_orders.columns] + \
               [c for c in work_orders.columns if c not in preferred_cols]
        work_orders = work_orders[cols]

        work_orders.to_csv(filename, index=False)
        return filename

    def generate_route_summary_csv(self, filename="route_summary.csv"):
        """One row per route with summary stats and endpoints (no crew)."""
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

# Example usage
if __name__ == "__main__":
    optimizer = PotholeRepairOptimizer("./detections_simulated.csv")
    
    # Step 1: Cluster potholes
    clustered_data = optimizer.cluster_potholes()
    print(f"Found {len(clustered_data['cluster'].unique())} clusters")
    
    # Step 2: Optimize routes
    routes, stats = optimizer.optimize_routes()
    print("\nRoute Statistics:")
    for cluster_id, stat in stats.items():
        print(f"Cluster {cluster_id}: {stat['num_potholes']} potholes, "
              f"{stat['total_km']}km, {stat['total_time_mins']}mins")
    
        # Step 3: Generate CSVs only (no HTML output)
    work_orders_csv = optimizer.generate_work_orders_csv('work_orders.csv')
    summary_csv = optimizer.generate_route_summary_csv('route_summary.csv')
    print(f"\nWork orders saved to {work_orders_csv}")
    print(f"Route summary saved to {summary_csv}")
