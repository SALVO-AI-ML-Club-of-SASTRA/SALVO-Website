# Generated by Django 5.1.4 on 2024-12-13 08:46

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tracker', '0006_member_joined_on_alter_meeting_start_time'),
    ]

    operations = [
        migrations.AlterField(
            model_name='meeting',
            name='start_time',
            field=models.TimeField(default=datetime.datetime(2024, 12, 13, 14, 16, 7, 984877)),
        ),
        migrations.AlterField(
            model_name='member',
            name='joined_on',
            field=models.DateField(default=datetime.datetime(2024, 12, 13, 8, 46, 7, 984877, tzinfo=datetime.timezone.utc)),
        ),
    ]
