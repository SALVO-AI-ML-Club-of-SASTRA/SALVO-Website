from django.urls import path
from . import views

urlpatterns = [
    path('', views.username_page, name='username'),
    path('draw/', views.draw_page, name='draw'),
    path('predict/', views.predict, name='predict'),  # AJAX endpoint
    path('models/', views.model_management, name='models'),  # NEW
    path('upload-model/', views.upload_model, name='upload_model'),
    path('get-models/', views.get_models, name='get_models'),
    path('delete-model/', views.delete_model, name='delete_model'),
    path('get-models/', views.get_models, name='get_models'),
 


]