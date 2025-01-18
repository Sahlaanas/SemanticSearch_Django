from django.urls import path
from . import views

urlpatterns = [
    
    path('', views.search, name='search'),
    path('add_faq/', views.add_faq, name='add_faq'),
    
    
    
]
