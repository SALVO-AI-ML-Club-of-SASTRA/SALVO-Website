from django.db import models
import datetime as dt
from django.utils import timezone


class Account(models.Model):
    """
        Public population of SASTRA will be given an account each
        Register Number is the primary key.
        Since a club member has to be a student of SASTRA, class Member inherits Account.
    """
    name = models.CharField(max_length=50)
    register_no = models.PositiveIntegerField(unique=True)
    sastra_email = models.EmailField()
    branch = models.CharField(max_length=100)
    batch = models.PositiveIntegerField()
    posts = models.CharField(max_length=1000000, default="[0]")
    password = models.CharField(max_length=20)
    # posts stores list of post_id as a json string. will dump and load whenever necessary.


class Member(models.Model):
    """
        Class for Permanent Members of SALVO.
        Roles = {Member, Coordinator, Lead}
        TO-DO: Find Formula for Contribution Score
        Privileges: Can Verify Posts, apart from posting.
    """
    name = models.CharField(max_length=50)
    register_no = models.PositiveIntegerField(unique=True)
    sastra_email = models.EmailField()
    branch = models.CharField(max_length=100)
    batch = models.PositiveIntegerField()
    posts = models.CharField(max_length=1000000, default="[0]")
    password = models.CharField(max_length=20)
    club_role = models.CharField(max_length=40)
    join_date = models.DateField(default=timezone.now)
    contribution_score = models.FloatField(default=0.0)
    attendance_percentage = models.FloatField(default=0.0)


class Post(models.Model):
    """
        Class for Storage of Posted Contents.
        Verification done by members only.
        Verified_by attribute points to regno of Member who verified.
        likes is an attribute to track like count.
        author_reg_no points to regno of Account that posts the post.
    """
    post_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=50)
    content = models.CharField(max_length=1000000000)
    author_reg_no = models.PositiveIntegerField()
    date = models.DateTimeField(default=dt.datetime.now())
    verified = models.BooleanField(default=False)
    verified_by = models.PositiveIntegerField(blank=True, null=True)
    likes = models.IntegerField(default=0)
    tags = models.JSONField(default=list)


class JoinRequest(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Accepted', 'Accepted'),
        ('Rejected', 'Rejected'),
    ]

    account = models.ForeignKey(Account, on_delete=models.CASCADE)
    reason_to_join = models.TextField()
    why_recruit = models.TextField()
    other_clubs = models.TextField()
    resume = models.FileField(upload_to='resumes/', null=True, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')
    upvotes = models.ManyToManyField(Member, blank=True)

class PostLike(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    register_no = models.PositiveIntegerField()  # from Account or Member

    class Meta:
        unique_together = ('post', 'register_no')
