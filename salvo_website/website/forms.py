from django import forms
from django.contrib.auth.hashers import make_password
from .models import Account, Member


class AccountRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Account
        fields = ['name', 'register_no', 'sastra_email', 'branch', 'batch', 'password']

    def save(self, commit=True):
        account = super().save(commit=False)
        account.password = make_password(self.cleaned_data['password'])
        if commit:
            account.save()
        return account


class MemberRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Member
        fields = ['name', 'register_no', 'sastra_email', 'branch', 'batch', 'password', 'club_role']

    def save(self, commit=True):
        member = super().save(commit=False)
        member.password = make_password(self.cleaned_data['password'])
        if commit:
            member.save()
        return member


class LoginForm(forms.Form):
    register_no = forms.IntegerField()
    password = forms.CharField(widget=forms.PasswordInput)