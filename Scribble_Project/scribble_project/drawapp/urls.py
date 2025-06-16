from django.urls import path
from . import views

urlpatterns = [
    path('', views.username_page, name='username'),
    path('draw/', views.draw_page, name='draw'),
    path('predict/', views.predict, name='predict'),  # AJAX endpoint
]