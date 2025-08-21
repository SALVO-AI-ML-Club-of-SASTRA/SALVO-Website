from django.contrib import admin
from website.models import Post,Account,Member

# Register your models here.
admin.site.register(Post)
admin.site.register(Account)
admin.site.register(Member)