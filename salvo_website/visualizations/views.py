# views.py - Decision Tree Visualization Views
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def decision_tree_page(request):
    """Main decision tree visualization page"""
    return render(request, 'visualizations/decision_tree.html')


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
