from django import forms
from django.contrib.auth.hashers import make_password
from .models import Account, Member, JoinRequest


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
    club_role = forms.ChoiceField(choices=[('Member','Member'),('Co-ordinator','Co-ordinator'),('Lead','Lead'),('Ex-Lead','Ex-Lead')])

    class Meta:
        model = Member
        fields = ['name', 'register_no', 'sastra_email', 'branch', 'batch', 'password', 'club_role']

    def save(self, commit=True):
        member = super().save(commit=False)
        member.password = make_password(self.cleaned_data['password'])
        member.club_role = self.cleaned_data['club_role']
        if commit:
            member.save()
        return member


class LoginForm(forms.Form):
    register_no = forms.IntegerField()
    password = forms.CharField(widget=forms.PasswordInput)


class JoinRequestForm(forms.ModelForm):
    class Meta:
        model = JoinRequest
        fields = ['reason_to_join', 'why_recruit', 'other_clubs', 'resume']
        widgets = {
            'reason_to_join': forms.Textarea(attrs={'rows': 3}),
            'why_recruit': forms.Textarea(attrs={'rows': 3}),
            'other_clubs': forms.Textarea(attrs={'rows': 2}),
        }
