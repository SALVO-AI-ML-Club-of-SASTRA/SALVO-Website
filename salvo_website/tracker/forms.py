from django import forms

class MemberForm(forms.Form):
    name=forms.CharField(max_length=50)
    emailid=forms.EmailField()
    regno=forms.IntegerField(max_value=999999999,min_value=100000000)
    role=forms.ChoiceField(choices=[('Member','Member'),('Co-ordinator','Co-ordinator'),('Lead','Lead')])

class AttendanceFileForm(forms.Form):
    meeting_code=forms.CharField(max_length=25)
    meeting_title=forms.CharField(max_length=100)
    meeting_date=forms.DateField(widget=forms.SelectDateWidget)
    file=forms.FileField(allow_empty_file=False)

class AddMinutes(forms.Form):
    minutes=forms.CharField(widget=forms.Textarea(attrs={'rows':20,'cols':20}))
    