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
]
