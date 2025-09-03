# urls.py - Visualizations URL Configuration
from django.urls import path
from . import views

app_name = 'visualizations'

urlpatterns = [
    # Main decision tree page
    path('', views.decision_tree_page, name='decision_tree'),
    path('decision-tree/', views.decision_tree_page, name='decision_tree_page'),
    
    # API endpoints
    path('api/status/', views.server_status, name='server_status'),
    path('api/build-tree/', views.build_tree_view, name='build_tree'),
    path('api/predict/', views.predict_view, name='predict'),
    path('api/get-defaults/', views.get_defaults_view, name='get_defaults'),
]
