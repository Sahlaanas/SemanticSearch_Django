# Generated by Django 5.1.5 on 2025-01-17 17:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='faq',
            name='answer_embedding',
            field=models.TextField(blank=True, null=True),
        ),
    ]
