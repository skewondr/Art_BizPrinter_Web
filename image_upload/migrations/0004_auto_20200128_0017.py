# Generated by Django 2.2.7 on 2020-01-28 00:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_upload', '0003_auto_20200127_2346'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='photo',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]
