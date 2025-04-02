from django.db import models
import datetime as dt


class Account:
    """
        Public population of SASTRA will be given an account each
        Register Number is the primary key.
        Since a club member has to be a student of SASTRA, class Member inherits Account.
    """
    name = models.CharField(max_length=50, required=True)
    register_no = models.PositiveIntegerField(max_length=9, unique=True)
    sastra_email = models.EmailField(required=True)
    branch = models.CharField(max_length=100,required=True)
    batch = models.PositiveIntegerField(max_length=4)
    posts = models.CharField(max_length=1000000, default="[0]")
    # posts stores list of post_id as a json string. will dump and load whenever necessary.

    def __init__(self,name,regno,branch,batch):
        self.name=name
        self.register_no=regno
        self.branch=branch
        self.batch=batch


class Member(Account):
    """
        Class for Permanent Members of SALVO.
        Roles = {Member, Coordinator, Lead}
        TO-DO: Find Formula for Contribution Score
        Privileges: Can Verify Posts, apart from posting.
    """
    club_role = models.CharField(max_length=40)
    join_date = models.DateField(default=dt.datetime.today())
    contribution_score = models.FloatField(default=0.0)
    attendance_percentage = models.FloatField(default=0.0)

    def __init__(self, name, regno, branch, batch, role):
        super().__init__(name,regno,branch,batch)
        self.role = role
        self.date = dt.datetime.today()

    def verify_post(self, post):
        post.verified = True


class Post:
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
    author_reg_no = models.PositiveIntegerField(max_length=9,unique=True)
    date = models.DateTimeField(default=dt.datetime.now())
    verified = models.BooleanField(default=False)
    verified_by = models.PositiveIntegerField(max_length=9)
    likes = models.IntegerField(default=0)

    def __init__(self,title,content,regno):
        self.title=title
        self.content=content
        self.author_reg_no=regno


