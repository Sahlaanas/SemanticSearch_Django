from django.db import models
from django.contrib.postgres.fields import ArrayField

class FAQ(models.Model):
    question = models.TextField()
    answer = models.TextField()
    answer_embedding = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.question
