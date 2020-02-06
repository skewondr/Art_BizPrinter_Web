from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User


class Photo(models.Model):
    photo = models.ImageField(null=True)
    label = models.IntegerField(default=1)
    converted = models.BooleanField(default=False)
