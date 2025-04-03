from django.shortcuts import render, redirect
from django.contrib.auth.hashers import check_password
from django.contrib import messages
from .models import Account, Member
from .forms import AccountRegistrationForm, MemberRegistrationForm, LoginForm


def home(request):
    return render(request, 'home.html')


def register_account(request):
    if request.method == 'POST':
        form = AccountRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Account registered successfully!")
            return redirect(login)
    else:
        form = AccountRegistrationForm()
    return render(request, 'register_account.html', {'form': form})


def login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            register_no = form.cleaned_data['register_no']
            password = form.cleaned_data['password']

            user = Account.objects.filter(register_no=register_no).first()
            member = Member.objects.filter(register_no=register_no).first()

            if user and check_password(password, user.password):
                request.session['user_type'] = 'account'
                request.session['register_no'] = user.register_no
                return redirect(account_dashboard)
            elif member and check_password(password, member.password):
                request.session['user_type'] = 'member'
                request.session['register_no'] = member.register_no
                return redirect(member_dashboard)
            else:
                messages.error(request, "Invalid credentials")
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


def register_member(request):
    if request.method == 'POST':
        form = MemberRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Member registered successfully!")
            return redirect(login)
    else:
        form = MemberRegistrationForm()
    return render(request, 'register_member.html', {'form': form})


def account_dashboard(request):
    if request.session.get('user_type') != 'account':
        return redirect(login)
    register_no = request.session.get('register_no')
    account = Account.objects.get(register_no=register_no)
    return render(request, 'account_dashboard.html', {'account': account})


def member_dashboard(request):
    if request.session.get('user_type') != 'member':
        return redirect(login)
    register_no = request.session.get('register_no')
    member = Member.objects.get(register_no=register_no)
    return render(request, 'member_dashboard.html', {'member': member})
