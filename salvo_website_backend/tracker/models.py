from django.db import models
import datetime as dt
import django.utils.timezone

# Create your models here.
class Member(models.Model):
    name=models.CharField(max_length=50,unique=True)
    emailid=models.EmailField()
    role=models.CharField(max_length=20,choices=[('Member','Member'),('Co-ordinator','Co-ordinator'),('Lead','Lead')])
    regno=models.IntegerField(unique=True)
    joined_on=models.DateField(default=django.utils.timezone.now())

    def __str__(self):
        return f"{self.name}-({self.role})"

class Meeting(models.Model):
    title=models.CharField(max_length=100)
    code=models.CharField(max_length=25,unique=True)
    date=models.DateField()
    start_time=models.TimeField(default=dt.datetime.now())
    end_time=models.TimeField()
    minutes_of_meeting=models.TextField(max_length=10000)
    attendees=models.TextField(max_length=1000)

    def __str__(self):
        return f"{self.title}-{self.code}"

class Attendance(models.Model):
    meeting_code=models.CharField(max_length=25)
    member_name=models.CharField(max_length=50)
    first_seen=models.DateTimeField()
    duration=models.DurationField()

class Contribution(models.Model):
    member_name=models.CharField(max_length=50,unique=True)
    file=models.FileField(upload_to=f'media/contrib/{member_name}')