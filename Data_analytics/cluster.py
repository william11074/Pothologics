import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint
import datetime

class PotholeRepairOptimizer:
    def __init__(self, filepath="pothole_dataset.csv"):
        self.df = pd.read_csv(filepath)
        self.cluster_routes = {}
        self.route_stats = {}
        self.cluster_polygons = {}
        
    def cluster_potholes(self, eps_km=0.2, min_samples=3):
        # cluster potholes using DBSCAN
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
        """Optimize repair routes for each cluster using greedy TSP approach"""
        for cluster_id in self.df['cluster'].unique():
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_points = self.df[self.df['cluster'] == cluster_id]
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
            self.cluster_routes[cluster_id] = cluster_points.iloc[route].copy()
            self.cluster_routes[cluster_id]['route_order'] = range(len(route))
            
            # Calculate route statistics
            total_distance = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            total_time = (total_distance / speed_kmh * 60) + (len(route) * repair_time_mins)
            
            self.route_stats[cluster_id] = {
                'num_potholes': len(route),
                'total_km': round(total_distance, 2),
                'total_time_mins': round(total_time),
                'speed_kmh': speed_kmh,
                'repair_time_mins': repair_time_mins
            }
            
        return self.cluster_routes, self.route_stats
    # Removed HTML visualization; this script now focuses on CSV outputs only.
    
    def generate_work_orders(self):
        """Generate work orders with route assignments"""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")
            
        work_orders = []
        for cluster_id, route_df in self.cluster_routes.items():
            route_df = route_df.copy()
            route_df['route_id'] = cluster_id
            route_df['crew_assignment'] = f"Crew {cluster_id % 3 + 1}"  # Assign to 3 crews
            work_orders.append(route_df)
            
        return pd.concat(work_orders).sort_values(['route_id', 'route_order'])
    def generate_work_orders_csv(self, filename="work_orders.csv"):
        """Write work orders to CSV (crew + route assignments + per-route stats)."""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")

        work_orders = []
        for cluster_id, route_df in self.cluster_routes.items():
            route_df = route_df.copy()
            route_df['route_id'] = cluster_id
            route_df['crew_assignment'] = f"Crew {cluster_id % 3 + 1}"

            stats = self.route_stats.get(cluster_id, {})
            route_df['total_km'] = stats.get('total_km')
            route_df['total_time_mins'] = stats.get('total_time_mins')
            route_df['speed_kmh'] = stats.get('speed_kmh')
            route_df['repair_time_mins'] = stats.get('repair_time_mins')

            work_orders.append(route_df)

        work_orders = pd.concat(work_orders, ignore_index=True)
        work_orders['crew_num'] = work_orders['crew_assignment'].str.extract(r'(\d+)').astype(int)
        work_orders = work_orders.sort_values(['crew_num', 'route_id', 'route_order']).drop(columns=['crew_num'])

        preferred_cols = [
            'route_id', 'crew_assignment', 'route_order',
            'latitude', 'longitude', 'severity', 'report_date',
            'total_km', 'total_time_mins', 'speed_kmh', 'repair_time_mins'
        ]
        cols = [c for c in preferred_cols if c in work_orders.columns] + \
               [c for c in work_orders.columns if c not in preferred_cols]
        work_orders = work_orders[cols]

        work_orders.to_csv(filename, index=False)
        return filename

    def generate_route_summary_csv(self, filename="route_summary.csv"):
        """One row per route with summary stats and endpoints."""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")

        rows = []
        for cluster_id, route_df in self.cluster_routes.items():
            stats = self.route_stats.get(cluster_id, {})
            crew = f"Crew {cluster_id % 3 + 1}"

            r = route_df.sort_values('route_order')
            start_lat, start_lon = r.iloc[0][['latitude', 'longitude']]
            end_lat, end_lon = r.iloc[-1][['latitude', 'longitude']]
            centroid_lat = r['latitude'].mean()
            centroid_lon = r['longitude'].mean()

            rows.append({
                'route_id': cluster_id,
                'crew_assignment': crew,
                'num_potholes': stats.get('num_potholes', len(r)),
                'total_km': stats.get('total_km'),
                'total_time_mins': stats.get('total_time_mins'),
                'speed_kmh': stats.get('speed_kmh'),
                'repair_time_mins': stats.get('repair_time_mins'),
                'start_lat': start_lat,
                'start_lon': start_lon,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
            })

        df = pd.DataFrame(rows)
        df['crew_num'] = df['crew_assignment'].str.extract(r'(\d+)').astype(int)
        df = df.sort_values(['crew_num', 'route_id']).drop(columns=['crew_num'])
        df.to_csv(filename, index=False)
        return filename

# Example usage
if __name__ == "__main__":
    optimizer = PotholeRepairOptimizer("pothole_dataset.csv")
    
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
