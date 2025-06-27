from django.contrib import admin
from .models import Attendance,Meeting,Member,Contribution
# Register your models here.
admin.site.register(Attendance)
admin.site.register(Meeting)
admin.site.register(Member)
admin.site.register(Contribution)