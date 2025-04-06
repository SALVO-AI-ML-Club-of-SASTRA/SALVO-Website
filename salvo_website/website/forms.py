from django import forms


class MemberSignupForm(forms.Form):
    name = forms.CharField(max_length=50)
    register_no = forms.IntegerField(min_value=100000000, max_value=300000000)
    sastra_email = forms.EmailField()
    branch = forms.CharField(max_length=100)
    batch = forms.IntegerField(min_value=2023)
    club_role = forms.CharField(max_length=40)
    password = forms.CharField(max_length=20)
    password2 = forms.CharField(max_length=20)


class AccountSignupForm(forms.Form):
    name = forms.CharField(max_length=50)
    register_no = forms.IntegerField(min_value=100000000, max_value=300000000)
    sastra_email = forms.EmailField()
    branch = forms.CharField(max_length=100)
    batch = forms.IntegerField(min_value=2023)
    password = forms.CharField(max_length=20)
    password2 = forms.CharField(max_length=20)

class LoginForm(forms.Form):
    register_no = forms.IntegerField(min_value=100000000, max_value=300000000)
    password = forms.CharField(max_length=20)

