from django.db import models

# Create your models here.
class Post(models.Model):
    IMG_B=models.CharField(max_length=200,null=True)
    IMG_A=models.CharField(max_length=200,null=True)

    def publish(self):
        self.published_date = timezone.now
        self.save()
