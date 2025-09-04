# urls.py - Visualizations URL Configuration
from django.urls import path
from . import views

app_name = 'visualizations'

urlpatterns = [
    # Main pages
    path('', views.visualizations_home, name='visualizations_home'),
    path('decision-tree/', views.decision_tree_page, name='decision_tree_page'),
    path('kmeans/', views.kmeans_page, name='kmeans_page'),
    path('dbscan/', views.dbscan_page, name='dbscan_page'),
    path('linear-regression/', views.linear_regression_page, name='linear_regression_page'),
    path('svm/', views.svm_page, name='svm_page'),
    
    # Decision Tree API endpoints
    path('api/status/', views.server_status, name='server_status'),
    path('api/build-tree/', views.build_tree_view, name='build_tree'),
    path('api/predict/', views.predict_view, name='predict'),
    path('api/get-defaults/', views.get_defaults_view, name='get_defaults'),
    
    # K-means API endpoints
    path('api/kmeans/elbow/', views.kmeans_elbow_method, name='kmeans_elbow'),
    path('api/kmeans/cluster/', views.kmeans_cluster, name='kmeans_cluster'),
    path('api/kmeans/add-point/', views.kmeans_add_point, name='kmeans_add_point'),
    path('api/kmeans/student-data/', views.kmeans_student_data, name='kmeans_student_data'),
    
    # DBSCAN API endpoints
    path('api/dbscan/cluster/', views.dbscan_cluster, name='dbscan_cluster'),
    path('api/dbscan/sample-data/', views.dbscan_sample_data, name='dbscan_sample_data'),
    
    # Linear Regression API endpoints
    path('api/regression/fit/', views.linear_regression_fit, name='linear_regression_fit'),
    path('api/regression/predict/', views.linear_regression_predict, name='linear_regression_predict'),
    path('api/regression/gradient-descent/', views.linear_regression_gradient_descent, name='linear_regression_gradient_descent'),
    path('api/regression/sample-data/', views.linear_regression_sample_data, name='linear_regression_sample_data'),
    path('api/regression/load-csv/', views.linear_regression_load_csv, name='linear_regression_load_csv'),
    
    # SVM API endpoints
    path('api/svm/train/', views.svm_train, name='svm_train'),
    path('api/svm/predict/', views.svm_predict, name='svm_predict'),
    path('api/svm/sample-data/', views.svm_sample_data, name='svm_sample_data'),
    path('api/svm/kernel-comparison/', views.svm_kernel_comparison, name='svm_kernel_comparison'),
]
