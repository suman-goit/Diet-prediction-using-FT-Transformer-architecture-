from django.urls import path

from . import views

urlpatterns = [
    path("auth/signup/", views.signup),
    path("auth/login/", views.login),
    path("auth/logout/", views.logout),
    path("auth/me/", views.me),

    path("assessment/", views.create_assessment),
    path("assessment/latest/", views.assessment_latest),
    path("assessment/history/", views.assessment_history),
    path("assessment/<int:pk>/", views.assessment_delete),

    path("meal-plan/generate/", views.generate_meal_plan),
    path("meal-plan/latest/", views.meal_plan_latest),
    path("meal-plan/regenerate-meal/", views.regenerate_meal),

    path("history/", views.recommendation_history_list),
    path("history/<int:pk>/", views.recommendation_history_detail),
    path("history/<int:pk>/reconstruct/", views.recommendation_history_reconstruct),
]
