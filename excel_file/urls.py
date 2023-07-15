from django.urls import path
from excel_file import views

urlpatterns = [
    path("", views.home, name="home"),
    path('analyze', views.analyze, name="analyze")
]
