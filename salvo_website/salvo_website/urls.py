"""
URL configuration for salvo_website project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from website.views import home, login, member_dashboard, account_dashboard, register_member, register_account

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('home/', home),
    path('member_home/',member_dashboard),
    path('account_home/',account_dashboard),
    path('login/', login),
    path('member_signup/', register_member),
    path('account_signup/', register_account),
]
