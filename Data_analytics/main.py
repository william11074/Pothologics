import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import folium
from shapely.geometry import MultiPoint
import datetime

class PotholeRepairOptimizer:
    def __init__(self, filepath="pothole_dataset.csv"):
        self.df = pd.read_csv(filepath)
        self.cluster_routes = {}
        self.route_stats = {}
        self.cluster_polygons = {}
        
    def cluster_potholes(self, eps_km=0.2, min_samples=3):
        """Cluster potholes using DBSCAN with haversine distance"""
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
    
    def visualize_routes(self, show_polygons=True):
        """Create interactive map with optimized routes"""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")
            
        center = self.df[['latitude', 'longitude']].mean().values.tolist()
        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles='Jawg.Dark',
            attr=' '  # This sets the attribution to a single space
        )
        
        # Subtle color palette for routes
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Create layer groups
        route_lines = folium.FeatureGroup(name='Repair Routes', show=True).add_to(m)
        route_markers = folium.FeatureGroup(name='Route Markers', show=True).add_to(m)
        polygon_group = folium.FeatureGroup(name='Cluster Polygons', show=show_polygons).add_to(m)
        noise_marker_layer = folium.FeatureGroup(name='Isolated Potholes', show=True).add_to(m)
        
        # Add noise points (outliers)
        noise_points = self.df[self.df['cluster'] == -1]
        for _, row in noise_points.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='#555555',
                fill=True,
                fill_color='#555555',
                fill_opacity=0.7,
                popup="Isolated pothole"
            ).add_to(noise_marker_layer)

        # Add cluster polygons if enabled
        if show_polygons:
            for cluster_id, poly in self.cluster_polygons.items():
                try:
                    bounds = [[p[0], p[1]] for p in poly.exterior.coords]
                    folium.Polygon(
                        locations=bounds,
                        color='#555555',
                        fill=True,
                        fill_color='#555555',
                        fill_opacity=0.1,
                        weight=1,
                        popup=f"Cluster {cluster_id}"
                    ).add_to(polygon_group)
                except:
                    pass
        
        # Add routes and markers
        for i, (cluster_id, route_df) in enumerate(self.cluster_routes.items()):
            color = colors[i % len(colors)]
            
            # Add route line
            folium.PolyLine(
                locations=route_df[['latitude', 'longitude']].values.tolist(),
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Route {cluster_id}<br>"
                      f"Potholes: {len(route_df)}<br>"
                      f"Distance: {self.route_stats[cluster_id]['total_km']}km<br>"
                      f"Time: {self.route_stats[cluster_id]['total_time_mins']}mins"
            ).add_to(route_lines)
            
            # Add markers with order numbers (smaller and subtle)
            for idx, row in route_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,  # Smaller markers
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Route {cluster_id}<br>"
                          f"Order: {row['route_order']+1}<br>"
                          f"Severity: {row.get('severity', 'N/A')}"
                ).add_to(route_markers)
        
        folium.LayerControl().add_to(m)
        return m
    
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
    
    def generate_work_orders_html(self, filename="work_orders.html"):
        """Generate styled HTML page with work orders"""
        work_orders = self.generate_work_orders()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pothole Repair Work Orders</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    color: #e2e8f0;
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px; 
                }}
                header {{ 
                    text-align: center; 
                    margin-bottom: 30px; 
                    border-bottom: 2px solid #eee; 
                    padding-bottom: 20px; 
                }}
                h1 {{ 
                    color: #e2e8f0; 
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 10px; 
                }}
                th {{ 
                    background: #2c3e50; 
                    color: white; 
                    text-align: left; 
                    padding: 12px; 
                }}
                tr:nth-child(even) {{ 
                    background: #f2f2f2; 
                }}
                td {{ 
                    padding: 10px 12px; 
                    border-bottom: 1px solid #ddd; 
                }}
                .route-card {{ 
                    background: #f9f9f9; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                    margin-bottom: 25px; 
                    overflow: hidden; 
                }}
                .route-header {{ 
                    background: #3498db; 
                    color: white; 
                    padding: 15px 20px; 
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center; 
                }}
                .route-stats {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px; 
                    padding: 20px; 
                    background: #e8f4fc; 
                }}
                .stat-box {{ 
                    background: white; 
                    padding: 12px; 
                    border-radius: 6px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
                }}
                .stat-box h3 {{ 
                    margin-top: 0; 
                    color: #2c3e50; 
                }}
                .pothole-list {{ 
                    padding: 20px; 
                }}
                .order-col {{ 
                    width: 50px; 
                    text-align: center; 
                }}
                .footer {{ 
                    text-align: center; 
                    margin-top: 30px; 
                    padding-top: 20px; 
                    border-top: 1px solid #eee; 
                    color: #7f8c8d; 
                    font-size: 0.9em; 
                }}
                @media print {{
                    .route-card {{ box-shadow: none; page-break-inside: avoid; }}
                    .route-header {{ background: #3498db !important; -webkit-print-color-adjust: exact; }}
                    th {{ background: #2c3e50 !important; -webkit-print-color-adjust: exact; }}
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Pothole Repair Work Orders</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </header>
        """
        
        # Group by crew for better organization
        work_orders['crew_num'] = work_orders['crew_assignment'].str.extract(r'(\d+)').astype(int)
        work_orders = work_orders.sort_values(['crew_num', 'route_id', 'route_order'])
        
        current_crew = None
        for crew in sorted(work_orders['crew_num'].unique()):
            crew_orders = work_orders[work_orders['crew_num'] == crew]
            html_content += f"<h2>Crew {crew} Assignments</h2>"
            
            for route_id in sorted(crew_orders['route_id'].unique()):
                route_df = crew_orders[crew_orders['route_id'] == route_id]
                route_stats = self.route_stats.get(route_id, {})
                
                html_content += f"""
                <div class="route-card">
                    <div class="route-header">
                        <h2>Route {route_id}</h2>
                        <div>Crew: {route_df.iloc[0]['crew_assignment']}</div>
                    </div>
                    
                    <div class="route-stats">
                        <div class="stat-box">
                            <h3>Route Summary</h3>
                            <p><strong>Potholes:</strong> {len(route_df)}</p>
                            <p><strong>Total Distance:</strong> {route_stats.get('total_km', 'N/A')} km</p>
                            <p><strong>Estimated Time:</strong> {route_stats.get('total_time_mins', 'N/A')} minutes</p>
                        </div>
                        
                        <div class="stat-box">
                            <h3>Repair Details</h3>
                            <p><strong>Travel Speed:</strong> {route_stats.get('speed_kmh', 'N/A')} km/h</p>
                            <p><strong>Repair Time:</strong> {route_stats.get('repair_time_mins', 'N/A')} min/pothole</p>
                        </div>
                    </div>
                    
                    <div class="pothole-list">
                        <h3>Repair Sequence</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th class="order-col">#</th>
                                    <th>Location</th>
                                    <th>Severity</th>
                                    <th>Road Type</th>
                                    <th>Report Date</th>
                                </tr>
                            </thead>
                            <tbody>
                """
                
                for _, row in route_df.iterrows():
                    html_content += f"""
                    <tr>
                        <td class="order-col">{row['route_order']+1}</td>
                        <td>{row['latitude']:.6f}, {row['longitude']:.6f}</td>
                        <td>{row.get('severity', 'N/A')}</td>
                        <td>{row.get('road_type', 'N/A')}</td>
                        <td>{row.get('report_date', 'N/A')}</td>
                    </tr>
                    """
                
                html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
                """
        
        html_content += f"""
            <div class="footer">
                <p>Total Potholes: {len(work_orders)} | Total Routes: {len(self.cluster_routes)}</p>
                <p>Generated by Pothole Repair Optimization System</p>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
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
    
    # Step 3: Generate work orders HTML
    html_file = optimizer.generate_work_orders_html('work_orders.html')
    print(f"\nWork orders saved to {html_file}")
    
    # Step 4: Visualize with polygon toggle
    map = optimizer.visualize_routes(show_polygons=True)
    map.save('pothole_repair_routes.html')
    print("\nRoute visualization saved to pothole_repair_routes.html")