# views.py - Decision Tree Visualization Views
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
import pandas as pd


def visualizations_home(request):
    """Main visualizations hub page"""
    return render(request, 'visualizations/home.html')


def decision_tree_page(request):
    """Main decision tree visualization page"""
    return render(request, 'visualizations/decision_tree.html')


def kmeans_page(request):
    """Main K-means clustering visualization page"""
    return render(request, 'visualizations/kmeans.html')


def dbscan_page(request):
    """Main DBSCAN clustering visualization page"""
    return render(request, 'visualizations/dbscan.html')


@csrf_exempt
def server_status(request):
    """Simple endpoint to check if server is online"""
    return JsonResponse({
        'status': 'online',
        'message': 'Django server is running'
    })


@csrf_exempt
def build_tree_view(request):
    """Django view to build decision tree"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dataset_name = data.get('dataset', 'iris')
            
            # Load dataset
            dataset_info = get_sample_data(dataset_name)
            
            # Build tree
            tree = build_decision_tree(
                dataset_info['data'], 
                dataset_info['target'],
                problem_type=dataset_info['problem_type'],
                max_depth=3,  # Limit depth for better visualization
                min_samples_leaf=2
            )
            
            # Convert to JSON
            tree_json = tree_to_json(
                tree, 
                dataset_info['feature_names'],
                problem_type=dataset_info['problem_type']
            )
            
            # Return response in format expected by frontend
            return JsonResponse({
                'tree': tree_json,
                'featureNames': dataset_info['feature_names'],
                'targetNames': dataset_info['target_names'],
                'problem_type': dataset_info['problem_type'],
                'dataset': dataset_name
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def predict_view(request):
    """Django view to make prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = data.get('features', {})
            feature_names = data.get('featureNames', [])
            dataset_name = data.get('dataset', 'iris')
            
            # Load dataset and build tree
            dataset_info = get_sample_data(dataset_name)
            tree = build_decision_tree(
                dataset_info['data'], 
                dataset_info['target'],
                problem_type=dataset_info['problem_type']
            )
            
            # Make prediction
            result = predict_with_tree(
                tree, 
                features, 
                feature_names,
                problem_type=dataset_info['problem_type']
            )
            
            # Add class name for classification problems
            if dataset_info['problem_type'] == 'classification' and 'prediction' in result:
                result['className'] = dataset_info['target_names'][result['prediction']]
            
            return JsonResponse(result)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def get_defaults_view(request):
    """API endpoint to get default values for a dataset"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dataset_name = data.get('dataset', 'iris')
            
            defaults = get_default_values(dataset_name)
            return JsonResponse({
                'defaults': defaults,
                'dataset': dataset_name
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


# Helper functions
def get_sample_data(dataset_name):
    """Load sample dataset"""
    try:
        if dataset_name == 'iris':
            data = load_iris()
            problem_type = 'classification'
        elif dataset_name == 'wine':
            data = load_wine()
            problem_type = 'classification'
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            problem_type = 'classification'
        elif dataset_name == 'digits':
            data = load_digits()
            problem_type = 'classification'
        elif dataset_name == 'diabetes':
            data = load_diabetes()
            problem_type = 'regression'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        feature_names = [name.replace(' ', '_').lower() for name in data.feature_names]
        
        if problem_type == 'classification':
            target_names = list(data.target_names)
        else:
            target_names = ['target_value']  # For regression
        
        return {
            'feature_names': feature_names,
            'target_names': target_names,
            'data': data.data.tolist(),
            'target': data.target.tolist(),
            'problem_type': problem_type
        }
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        raise


def build_decision_tree(X, y, problem_type='classification', max_depth=None, min_samples_leaf=1):
    """Build and return a decision tree classifier or regressor"""
    try:
        if problem_type == 'classification':
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:  # regression
            clf = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
        clf.fit(X, y)
        return clf
    except Exception as e:
        print(f"Error building decision tree: {str(e)}")
        raise


def tree_to_json(tree, feature_names, problem_type='classification'):
    """Convert sklearn tree to JSON representation with node IDs for frontend"""
    try:
        tree_ = tree.tree_
        node_counter = 0  # Counter to assign unique IDs
        
        def recurse(node, depth=0):
            nonlocal node_counter
            current_id = node_counter
            node_counter += 1
            
            if tree_.feature[node] != -2:  # Not a leaf node
                feature_index = tree_.feature[node]
                if feature_index < len(feature_names):
                    feature_name = feature_names[feature_index]
                else:
                    feature_name = f"feature_{feature_index}"
                
                # Recursively build left and right subtrees
                left_child = recurse(tree_.children_left[node], depth + 1)
                right_child = recurse(tree_.children_right[node], depth + 1)
                
                return {
                    'id': current_id,
                    'type': 'decision',
                    'feature': feature_name,
                    'threshold': float(tree_.threshold[node]),
                    'left': left_child,
                    'right': right_child,
                    'samples': int(tree_.n_node_samples[node]),
                    'impurity': float(tree_.impurity[node]),
                    'value': tree_.value[node].tolist()[0] if hasattr(tree_.value[node], 'tolist') else []
                }
            else:  # Leaf node
                if problem_type == 'classification':
                    return {
                        'id': current_id,
                        'type': 'leaf',
                        'value': tree_.value[node].tolist()[0] if hasattr(tree_.value[node], 'tolist') else [],
                        'class': int(np.argmax(tree_.value[node])),
                        'samples': int(tree_.n_node_samples[node]),
                        'impurity': float(tree_.impurity[node])
                    }
                else:  # regression
                    return {
                        'id': current_id,
                        'type': 'leaf',
                        'value': float(tree_.value[node][0][0]),
                        'samples': int(tree_.n_node_samples[node]),
                        'impurity': float(tree_.impurity[node])
                    }
        
        result = recurse(0)
        return result
        
    except Exception as e:
        print(f"Error converting tree to JSON: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a simple fallback tree
        return {
            'id': 0,
            'type': 'decision',
            'feature': 'fallback',
            'threshold': 0.5,
            'samples': 100,
            'value': [50, 50],
            'impurity': 0.5,
            'left': {
                'id': 1,
                'type': 'leaf',
                'class': 0,
                'samples': 50,
                'value': [50, 0],
                'impurity': 0.0
            },
            'right': {
                'id': 2,
                'type': 'leaf',
                'class': 1,
                'samples': 50,
                'value': [0, 50],
                'impurity': 0.0
            }
        }


def predict_with_tree(tree, features, feature_names, problem_type='classification'):
    """Make a prediction using the decision tree"""
    try:
        # Convert features to the correct order based on feature_names
        ordered_features = []
        for name in feature_names:
            # Handle different feature name formats
            clean_name = name.replace(' ', '_').lower()
            if clean_name in features:
                ordered_features.append(float(features[clean_name]))
            elif name in features:
                ordered_features.append(float(features[name]))
            else:
                # Try to find a matching feature name
                found = False
                for key in features.keys():
                    if key.replace(' ', '_').lower() == clean_name:
                        ordered_features.append(float(features[key]))
                        found = True
                        break
                if not found:
                    ordered_features.append(0.0)  # Default value if feature missing
        
        if problem_type == 'classification':
            prediction = tree.predict([ordered_features])[0]
            probabilities = tree.predict_proba([ordered_features])[0]
            
            return {
                'prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'className': str(prediction)  # Will be mapped to actual name in frontend
            }
        else:  # regression
            prediction = tree.predict([ordered_features])[0]
            
            return {
                'prediction': float(prediction),
                'probabilities': [1.0],  # Not applicable for regression
                'value': float(prediction)
            }
            
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise


def get_default_values(dataset_name):
    """Return default values for each dataset to pre-fill the prediction form"""
    if dataset_name == 'iris':
        return {
            'sepal_length_cm': 5.0,
            'sepal_width_cm': 3.5,
            'petal_length_cm': 1.5,
            'petal_width_cm': 0.2
        }
    elif dataset_name == 'wine':
        return {
            'alcohol': 12.0,
            'malic_acid': 2.0,
            'ash': 2.5,
            'alcalinity_of_ash': 18.0,
            'magnesium': 95.0,
            'total_phenols': 2.5,
            'flavanoids': 2.0,
            'nonflavanoid_phenols': 0.3,
            'proanthocyanins': 1.5,
            'color_intensity': 4.0,
            'hue': 1.0,
            'od280_od315_of_diluted_wines': 2.5,
            'proline': 800.0
        }
    elif dataset_name == 'breast_cancer':
        return {
            'mean_radius': 15.0,
            'mean_texture': 20.0,
            'mean_perimeter': 100.0,
            'mean_area': 700.0,
            'mean_smoothness': 0.1,
            'mean_compactness': 0.15,
            'mean_concavity': 0.1,
            'mean_concave_points': 0.05,
            'mean_symmetry': 0.2,
            'mean_fractal_dimension': 0.06
        }
    elif dataset_name == 'digits':
        # For digits, we typically use images, but we'll provide some default values
        return {f'pixel_{i}_{j}': 5.0 for i in range(8) for j in range(8)}  # 8x8 subset
    elif dataset_name == 'diabetes':
        return {
            'age': 0.04,
            'sex': 0.05,
            'bmi': 0.06,
            'bp': 0.02,
            's1': -0.01,
            's2': -0.04,
            's3': -0.04,
            's4': -0.00,
            's5': 0.02,
            's6': -0.02
        }
    else:
        return {}


# ===========================
# K-MEANS CLUSTERING ENDPOINTS
# ===========================

@csrf_exempt
def kmeans_elbow_method(request):
    """API endpoint to calculate elbow method for optimal K"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dataset_type = data.get('dataset_type', 'generated')
            n_points = data.get('n_points', 100)
            
            print(f"Generating data: type={dataset_type}, points={n_points}")  # Debug log
            
            # Generate or use predefined data
            if dataset_type == 'generated':
                X = generate_sample_data(n_points)
            elif dataset_type == 'student':
                X = get_student_sample_data()
            else:
                X = get_blob_data()
            
            print(f"Generated data shape: {X.shape}")  # Debug log
            
            # Calculate WCSS for different K values
            wcss = []
            k_range = range(1, min(11, len(X)))
            
            for k in k_range:
                if k <= len(X):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)
                else:
                    wcss.append(0)
            
            print(f"WCSS calculated: {wcss}")  # Debug log
            
            return JsonResponse({
                'k_values': list(k_range),
                'wcss': wcss,
                'data_points': X.tolist(),
                'optimal_k': find_optimal_k(wcss)
            })
            
        except Exception as e:
            print(f"Error in kmeans_elbow_method: {e}")  # Debug log
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def kmeans_cluster(request):
    """API endpoint to perform K-means clustering"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            X = np.array(data.get('data_points', []))
            k = data.get('k', 3)
            animate_steps = data.get('animate_steps', False)
            
            if len(X) == 0:
                raise ValueError("No data points provided")
            
            if animate_steps:
                # Return step-by-step clustering animation data
                steps = perform_kmeans_animation(X, k)
                return JsonResponse({
                    'steps': steps,
                    'k': k,
                    'data_points': X.tolist()
                })
            else:
                # Return final clustering result
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                centroids = kmeans.cluster_centers_
                
                # Calculate cluster performance (if 2D, use distance from origin as "performance")
                cluster_performance = calculate_cluster_performance(X, labels, k)
                
                return JsonResponse({
                    'labels': labels.tolist(),
                    'centroids': centroids.tolist(),
                    'cluster_performance': cluster_performance,
                    'inertia': kmeans.inertia_,
                    'k': k
                })
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt 
def kmeans_add_point(request):
    """API endpoint to add a point to the clustering"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            x = data.get('x', 0)
            y = data.get('y', 0)
            
            return JsonResponse({
                'point': [x, y],
                'status': 'added'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def kmeans_student_data(request):
    """API endpoint to get sample student data"""
    if request.method == 'GET':
        try:
            student_data = get_student_sample_data()
            return JsonResponse({
                'data': student_data.tolist(),
                'labels': ['Math', 'Physics', 'Chemistry'],
                'student_names': [f'Student_{i+1}' for i in range(len(student_data))]
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


# Helper functions for K-means clustering

def generate_sample_data(n_points=100):
    """Generate sample 2D data for clustering"""
    np.random.seed(42)
    
    # Create 3 clusters manually
    cluster1 = np.random.normal([20, 20], [5, 5], (n_points//3, 2))
    cluster2 = np.random.normal([60, 20], [8, 6], (n_points//3, 2))
    cluster3 = np.random.normal([40, 60], [6, 8], (n_points//3, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Add some noise points
    noise = np.random.uniform(0, 80, (n_points//10, 2))
    data = np.vstack([data, noise])
    
    return data


def get_student_sample_data():
    """Generate sample student academic data (Math, Physics, Chemistry)"""
    np.random.seed(42)
    n_students = 30
    
    # High performers
    high_performers = np.random.normal([85, 80, 82], [5, 6, 5], (n_students//3, 3))
    
    # Medium performers  
    medium_performers = np.random.normal([65, 62, 68], [8, 7, 6], (n_students//3, 3))
    
    # Low performers
    low_performers = np.random.normal([45, 42, 48], [6, 8, 7], (n_students//3, 3))
    
    data = np.vstack([high_performers, medium_performers, low_performers])
    
    # Ensure marks are between 0-100
    data = np.clip(data, 0, 100)
    
    return data


def get_blob_data():
    """Generate blob-like data for clustering"""
    try:
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=100, centers=4, n_features=2, 
                          random_state=42, cluster_std=1.5)
        return X
    except ImportError:
        # Fallback to manual blob generation
        return generate_sample_data(100)


def find_optimal_k(wcss):
    """Find optimal K using elbow method"""
    if len(wcss) < 3:
        return 2
    
    # Calculate differences
    differences = []
    for i in range(1, len(wcss)):
        differences.append(wcss[i-1] - wcss[i])
    
    # Find the elbow point (where improvement starts to decrease significantly)
    max_diff_index = np.argmax(differences)
    
    # Optimal K is usually the point where the rate of decrease slows down
    optimal_k = max_diff_index + 2  # +2 because we start from k=1 and index from 0
    
    return min(optimal_k, len(wcss))


def perform_kmeans_animation(X, k):
    """Perform K-means step by step for animation"""
    steps = []
    max_iterations = 10
    
    # Initialize centroids randomly
    np.random.seed(42)
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)].copy()
    
    # Initial step
    steps.append({
        'iteration': 0,
        'centroids': centroids.tolist(),
        'assignments': [-1] * len(X),
        'phase': 'initialization'
    })
    
    for iteration in range(max_iterations):
        # Assignment step
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        assignments = np.argmin(distances, axis=0)
        
        steps.append({
            'iteration': iteration + 1,
            'centroids': centroids.tolist(),
            'assignments': assignments.tolist(),
            'phase': 'assignment'
        })
        
        # Update step
        new_centroids = np.array([X[assignments == i].mean(axis=0) for i in range(k)])
        
        # Handle empty clusters
        for i in range(k):
            if not np.any(assignments == i):
                new_centroids[i] = centroids[i]
        
        steps.append({
            'iteration': iteration + 1,
            'centroids': new_centroids.tolist(),
            'assignments': assignments.tolist(),
            'phase': 'update'
        })
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            steps.append({
                'iteration': iteration + 1,
                'centroids': new_centroids.tolist(),
                'assignments': assignments.tolist(),
                'phase': 'converged'
            })
            break
            
        centroids = new_centroids
    
    return steps


def calculate_cluster_performance(X, labels, k):
    """Calculate performance labels for clusters"""
    cluster_performance = {}
    
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            if X.shape[1] == 2:
                # For 2D data, use distance from origin as performance metric
                avg_distance = np.mean(np.linalg.norm(cluster_points, axis=1))
                cluster_performance[i] = {
                    'avg_performance': float(avg_distance),
                    'size': len(cluster_points)
                }
            elif X.shape[1] == 3:
                # For 3D data (like student marks), use average of all features
                avg_performance = np.mean(cluster_points)
                cluster_performance[i] = {
                    'avg_performance': float(avg_performance),
                    'size': len(cluster_points)
                }
            else:
                cluster_performance[i] = {
                    'avg_performance': 0.0,
                    'size': len(cluster_points)
                }
        else:
            cluster_performance[i] = {
                'avg_performance': 0.0,
                'size': 0
            }
    
    # Label clusters as High/Medium/Low based on performance
    performances = [cluster_performance[i]['avg_performance'] for i in range(k)]
    if len(performances) >= 3:
        sorted_indices = np.argsort(performances)
        labels_map = {}
        labels_map[sorted_indices[-1]] = 'High'
        labels_map[sorted_indices[0]] = 'Low'
        for i in range(k):
            if i not in labels_map:
                labels_map[i] = 'Medium'
    else:
        # For k < 3, just use High/Low
        sorted_indices = np.argsort(performances)
        labels_map = {}
        if len(sorted_indices) >= 2:
            labels_map[sorted_indices[-1]] = 'High'
            labels_map[sorted_indices[0]] = 'Low'
        else:
            labels_map[0] = 'Medium'
    
    # Add performance labels
    for i in range(k):
        cluster_performance[i]['label'] = labels_map.get(i, 'Medium')
    
    return cluster_performance


# ===========================
# DBSCAN CLUSTERING ENDPOINTS
# ===========================

@csrf_exempt
def dbscan_cluster(request):
    """API endpoint to perform DBSCAN clustering with step-by-step visualization"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            points = data.get('points', [])
            epsilon = float(data.get('epsilon', 0.5))
            min_pts = int(data.get('min_pts', 3))
            return_steps = data.get('return_steps', True)
            
            if not points:
                return JsonResponse({'error': 'No points provided'}, status=400)
            
            print(f"DBSCAN clustering: {len(points)} points, eps={epsilon}, min_pts={min_pts}")
            
            if return_steps:
                # Return step-by-step animation data
                steps = perform_dbscan_animation(points, epsilon, min_pts)
                return JsonResponse({
                    'steps': steps,
                    'epsilon': epsilon,
                    'min_pts': min_pts,
                    'points': points
                })
            else:
                # Return final clustering result
                from sklearn.cluster import DBSCAN
                import numpy as np
                
                X = np.array([[p['x'], p['y']] for p in points])
                
                # Scale epsilon to match coordinate system
                scaled_epsilon = epsilon * 100  # Convert to canvas coordinates
                
                dbscan = DBSCAN(eps=scaled_epsilon, min_samples=min_pts)
                labels = dbscan.fit_predict(X)
                
                # Calculate cluster statistics
                cluster_stats = calculate_dbscan_cluster_stats(X, labels)
                
                return JsonResponse({
                    'labels': labels.tolist(),
                    'cluster_stats': cluster_stats,
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'n_noise': list(labels).count(-1),
                    'epsilon': epsilon,
                    'min_pts': min_pts
                })
                
        except Exception as e:
            print(f"Error in dbscan_cluster: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def dbscan_sample_data(request):
    """API endpoint to get sample data for DBSCAN demonstration"""
    if request.method == 'GET':
        try:
            # Generate sample data with clusters and noise
            sample_points = generate_dbscan_sample_data()
            return JsonResponse({
                'points': sample_points,
                'recommended_epsilon': 0.3,
                'recommended_min_pts': 3
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


# Helper functions for DBSCAN clustering

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def perform_dbscan_animation(points, epsilon, min_pts):
    """Perform DBSCAN step by step for animation"""
    steps = []
    n_points = len(points)
    
    # Convert points to coordinate arrays
    coordinates = [(p['x'], p['y']) for p in points]
    
    # Scale epsilon to canvas coordinates
    scaled_epsilon = epsilon * 100
    
    # Initialize variables
    cluster_id = 0
    labels = [-1] * n_points
    visited = [False] * n_points
    point_types = ['unknown'] * n_points  # 'core', 'border', 'noise'
    
    # Initial step
    steps.append({
        'step': 0,
        'description': 'Starting DBSCAN algorithm',
        'labels': labels.copy(),
        'point_types': point_types.copy(),
        'current_point': None,
        'neighbors': [],
        'phase': 'initialization'
    })
    
    step_count = 1
    
    # Process each unvisited point
    for i in range(n_points):
        if visited[i]:
            continue
            
        visited[i] = True
        
        # Find neighbors within epsilon
        neighbors = []
        for j in range(n_points):
            if i != j and euclidean_distance(coordinates[i], coordinates[j]) <= scaled_epsilon:
                neighbors.append(j)
        
        # Show current point being examined
        steps.append({
            'step': step_count,
            'description': f'Examining point {i}: found {len(neighbors)} neighbors',
            'labels': labels.copy(),
            'point_types': point_types.copy(),
            'current_point': i,
            'neighbors': neighbors.copy(),
            'phase': 'examining'
        })
        step_count += 1
        
        # Check if it's a core point
        if len(neighbors) < min_pts:
            # Mark as noise (for now)
            point_types[i] = 'noise'
            steps.append({
                'step': step_count,
                'description': f'Point {i} marked as noise (insufficient neighbors)',
                'labels': labels.copy(),
                'point_types': point_types.copy(),
                'current_point': i,
                'neighbors': [],
                'phase': 'noise'
            })
            step_count += 1
        else:
            # Core point - start new cluster
            cluster_id += 1
            labels[i] = cluster_id
            point_types[i] = 'core'
            
            steps.append({
                'step': step_count,
                'description': f'Point {i} is core point - starting cluster {cluster_id}',
                'labels': labels.copy(),
                'point_types': point_types.copy(),
                'current_point': i,
                'neighbors': neighbors.copy(),
                'phase': 'core'
            })
            step_count += 1
            
            # Expand cluster using queue
            queue = neighbors.copy()
            
            while queue:
                current_neighbor = queue.pop(0)
                
                # If not visited, mark as visited and check
                if not visited[current_neighbor]:
                    visited[current_neighbor] = True
                    
                    # Find neighbors of this neighbor
                    neighbor_neighbors = []
                    for k in range(n_points):
                        if k != current_neighbor and euclidean_distance(coordinates[current_neighbor], coordinates[k]) <= scaled_epsilon:
                            neighbor_neighbors.append(k)
                    
                    steps.append({
                        'step': step_count,
                        'description': f'Expanding to point {current_neighbor}: found {len(neighbor_neighbors)} neighbors',
                        'labels': labels.copy(),
                        'point_types': point_types.copy(),
                        'current_point': current_neighbor,
                        'neighbors': neighbor_neighbors.copy(),
                        'phase': 'expanding'
                    })
                    step_count += 1
                    
                    # If this neighbor is also a core point, add its neighbors to queue
                    if len(neighbor_neighbors) >= min_pts:
                        point_types[current_neighbor] = 'core'
                        queue.extend([n for n in neighbor_neighbors if labels[n] == -1])
                
                # Add to current cluster if not already assigned
                if labels[current_neighbor] == -1:
                    labels[current_neighbor] = cluster_id
                    if point_types[current_neighbor] == 'unknown':
                        point_types[current_neighbor] = 'border'
                    
                    steps.append({
                        'step': step_count,
                        'description': f'Adding point {current_neighbor} to cluster {cluster_id}',
                        'labels': labels.copy(),
                        'point_types': point_types.copy(),
                        'current_point': current_neighbor,
                        'neighbors': [],
                        'phase': 'assign'
                    })
                    step_count += 1
    
    # Final step
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = labels.count(-1)
    
    steps.append({
        'step': step_count,
        'description': f'DBSCAN complete: {n_clusters} clusters, {n_noise} noise points',
        'labels': labels.copy(),
        'point_types': point_types.copy(),
        'current_point': None,
        'neighbors': [],
        'phase': 'complete'
    })
    
    return steps


def calculate_dbscan_cluster_stats(X, labels):
    """Calculate statistics for DBSCAN clusters"""
    unique_labels = set(labels)
    stats = {}
    
    for label in unique_labels:
        if label == -1:
            # Noise points
            noise_count = list(labels).count(-1)
            stats['noise'] = {
                'label': 'Noise',
                'count': noise_count,
                'color': 'gray'
            }
        else:
            # Regular cluster
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0)
            
            stats[f'cluster_{label}'] = {
                'label': f'Cluster {label}',
                'count': len(cluster_points),
                'centroid': centroid.tolist(),
                'color': get_cluster_color(label)
            }
    
    return stats


def get_cluster_color(cluster_id):
    """Get color for cluster visualization"""
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', 
              '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43']
    return colors[cluster_id % len(colors)]


def generate_dbscan_sample_data():
    """Generate sample data points for DBSCAN demonstration"""
    np.random.seed(42)
    points = []
    
    # Dense cluster 1 (top-left)
    cluster1_x = np.random.normal(150, 25, 15)
    cluster1_y = np.random.normal(150, 25, 15)
    
    # Dense cluster 2 (top-right) 
    cluster2_x = np.random.normal(350, 30, 20)
    cluster2_y = np.random.normal(150, 20, 20)
    
    # Dense cluster 3 (bottom-center)
    cluster3_x = np.random.normal(250, 35, 18)
    cluster3_y = np.random.normal(350, 30, 18)
    
    # Add some noise points
    noise_x = np.random.uniform(50, 450, 8)
    noise_y = np.random.uniform(50, 450, 8)
    
    # Combine all points
    all_x = np.concatenate([cluster1_x, cluster2_x, cluster3_x, noise_x])
    all_y = np.concatenate([cluster1_y, cluster2_y, cluster3_y, noise_y])
    
    # Ensure points are within canvas bounds
    all_x = np.clip(all_x, 10, 502)
    all_y = np.clip(all_y, 10, 502)
    
    # Convert to point format
    for x, y in zip(all_x, all_y):
        points.append({'x': int(x), 'y': int(y)})
    
    return points
