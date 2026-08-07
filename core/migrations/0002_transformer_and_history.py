from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name="userprofile",
            name="target_sugar_g",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="target_sodium_mg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="target_cholesterol_mg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="target_iron_mg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="target_calcium_mg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="target_vitamin_c_mg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.CreateModel(
            name="RecommendationHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True,
                                            serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("profile_snapshot", models.JSONField()),
                ("disease_predictions", models.JSONField()),
                ("constraint_engine_output", models.JSONField()),
                ("full_meal_plan", models.JSONField()),
                ("ilp_results", models.JSONField()),
                ("meal_plan", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE,
                                                    related_name="history_record", to="core.mealplan")),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                            related_name="recommendation_history",
                                            to=settings.AUTH_USER_MODEL)),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
