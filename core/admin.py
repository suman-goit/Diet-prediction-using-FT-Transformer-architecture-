from django.contrib import admin

from .models import (DayPlan, Food, Meal, MealItem, MealPlan,
                      RecommendationHistory, UserProfile)

admin.site.register(Food)
admin.site.register(UserProfile)
admin.site.register(MealPlan)
admin.site.register(DayPlan)
admin.site.register(Meal)
admin.site.register(MealItem)
admin.site.register(RecommendationHistory)
