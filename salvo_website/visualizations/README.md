# Decision Tree Visualizer

This Django app provides an interactive decision tree visualization system that has been integrated into the SALVO website while maintaining its AI-themed aesthetic.

## Features

### Backend (Django + scikit-learn)
- **Multiple Datasets**: Support for Iris, Wine, Breast Cancer, Digits, and Diabetes datasets
- **Real-time Training**: Decision trees are built on-demand with configurable parameters
- **API Endpoints**: RESTful API for tree building, prediction, and configuration
- **Error Handling**: Comprehensive error handling and fallback mechanisms

### Frontend (Vanilla JS + SALVO Theme)
- **Interactive Visualization**: SVG-based tree rendering with hover effects
- **Prediction Interface**: Dynamic form generation based on dataset features
- **Real-time Feedback**: Live prediction results with confidence scores
- **Responsive Design**: Mobile-friendly layout with SALVO's neural network theme

## File Structure

```
visualizations/
├── __init__.py
├── apps.py
├── models.py                    # No models needed (in-memory processing)
├── views.py                     # Core logic and API endpoints
├── urls.py                      # URL routing
├── admin.py
├── tests.py
├── migrations/
│   └── __init__.py
├── templates/
│   └── visualizations/
│       └── decision_tree.html   # Main visualization template
└── README.md                    # This file
```

## API Endpoints

### Main Page
- `GET /visualizations/` - Main decision tree visualization page
- `GET /visualizations/decision-tree/` - Alternative URL

### API Routes  
- `GET /visualizations/api/status/` - Check server status
- `POST /visualizations/api/build-tree/` - Build decision tree for dataset
- `POST /visualizations/api/predict/` - Make prediction with features
- `POST /visualizations/api/get-defaults/` - Get default feature values

## Usage

### Building a Decision Tree

1. **Select Dataset**: Choose from available datasets (Iris, Wine, etc.)
2. **Build Tree**: Click "Build Decision Tree" to train the model
3. **View Visualization**: Interactive SVG tree appears with nodes and branches

### Making Predictions

1. **Enter Features**: Fill in feature values (or use example values)
2. **Predict**: Click "Make Prediction" to get results
3. **View Results**: See predicted class/value with confidence scores
4. **Path Highlighting**: Visual path through tree for current prediction

## Technical Details

### Dataset Processing
```python
def get_sample_data(dataset_name):
    # Loads scikit-learn datasets
    # Normalizes feature names  
    # Returns structured data for frontend
```

### Tree Building
```python
def build_decision_tree(X, y, problem_type='classification', max_depth=None):
    # Uses DecisionTreeClassifier/Regressor
    # Configurable parameters for visualization
    # Returns trained model
```

### Tree Conversion
```python
def tree_to_json(tree, feature_names, problem_type='classification'):
    # Recursively converts sklearn tree to JSON
    # Assigns unique IDs to nodes
    # Handles both classification and regression
```

## Supported Datasets

| Dataset | Type | Features | Classes/Target |
|---------|------|----------|----------------|
| Iris | Classification | 4 | 3 species |
| Wine | Classification | 13 | 3 wine types |
| Breast Cancer | Classification | 30 | 2 (malignant/benign) |
| Digits | Classification | 64 | 10 digits |
| Diabetes | Regression | 10 | Continuous value |

## SALVO Theme Integration

The visualization maintains SALVO's AI-themed design:

### Color Scheme
- **Primary Cyan**: `#00D4FF` - Main accents and highlights
- **Accent Green**: `#00FF88` - Success states and active elements  
- **Neural Orange**: `#FF6B35` - Error states and warnings
- **Dark Background**: `#0A0F1C` - Main background
- **Card Background**: `rgba(15, 25, 45, 0.9)` - Content panels

### Typography
- **JetBrains Mono**: Code and technical elements
- **Poppins**: General text and UI elements

### Visual Effects
- Neural network background animation
- Gradient borders and glowing effects
- Smooth transitions and hover states
- Code-style syntax highlighting

## Future Enhancements

1. **Additional Algorithms**
   - Random Forest visualization
   - Linear Regression plotting  
   - Neural Network architecture
   - SVM decision boundaries

2. **Interactive Features**
   - Dataset upload functionality
   - Hyperparameter tuning interface
   - Model comparison tools
   - Export visualization options

3. **Performance Optimization**
   - Caching for repeated requests
   - Asynchronous processing
   - Progressive tree rendering
   - WebGL acceleration

## Dependencies

### Backend
- Django 5.1.4
- djangorestframework 3.14.0
- django-cors-headers 4.2.0
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy (via scikit-learn)

### Frontend
- Vanilla JavaScript (ES6+)
- SVG for tree visualization
- Bootstrap 5.3.2 (inherited from SALVO)
- Font Awesome 6.4.0 (inherited from SALVO)

## Development Notes

### Code Style
- Follows Django conventions
- PEP 8 compliant Python code
- ES6+ JavaScript with consistent formatting
- Semantic HTML5 structure

### Performance Considerations
- Trees limited to max_depth=3 for visualization clarity
- Feature inputs limited to first 8 for UI cleanliness  
- Error handling prevents server crashes
- Graceful degradation for missing dependencies

### Security
- CSRF protection enabled
- CORS configured for development
- Input validation on all endpoints
- No user data persistence (stateless operation)

This implementation successfully replicates the original DecisionTree module functionality while seamlessly integrating with the SALVO website's existing architecture and design language.
