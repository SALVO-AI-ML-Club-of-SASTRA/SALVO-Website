from django.shortcuts import render,redirect
from website.models import Account,Member,Post
from website.forms import LoginForm,MemberSignupForm,AccountSignupForm

# Create your views here.


def home(request):
    return render(request,'home.html')


def member_home(request,member):
    return render(request,'memberhome.html')


def account_home(request,account):
    return render(request,'accounthome.html')


def member_signup(request):
    """
    reinforce this function with a list of all current members so that everyone can't register as a member.
    """
    msg = ''
    mem_form = MemberSignupForm()
    if request.method == 'POST':
        mem_form = MemberSignupForm(request.POST)
        if mem_form.is_valid():
            name = mem_form.cleaned_data['name']
            regno = mem_form.cleaned_data['register_no']
            sastra_email = mem_form.cleaned_data['sastra_email']
            branch = mem_form.cleaned_data['branch']
            batch = mem_form.cleaned_data['batch']
            club_role = mem_form.cleaned_data['club_role']
            password = mem_form.cleaned_data['password']
            password2 = mem_form.cleaned_data['password2']
            if password == password2:
                mem = Member(name=name, register_no=regno, sastra_email=sastra_email, branch=branch, batch=batch, club_role = club_role)
                mem.save()
                msg = 'Member Account created Successfully :)'
            else:
                msg = 'Passwords do not match, Please Re-enter them.'
                return render(request, 'member_signup.html', {'msg': msg, 'form': mem_form})
            return redirect(login(request))
    return render(request, 'member_signup.html', {'msg': msg, 'form': mem_form})


def account_signup(request):
    msg=''
    acc_form=AccountSignupForm()
    if request.method == 'POST':
        acc_form=AccountSignupForm(request.POST)
        if acc_form.is_valid():
            name = acc_form.cleaned_data['name']
            regno = acc_form.cleaned_data['register_no']
            sastra_email = acc_form.cleaned_data['sastra_email']
            branch = acc_form.cleaned_data['branch']
            batch = acc_form.cleaned_data['batch']
            password = acc_form.cleaned_data['password']
            password2 = acc_form.cleaned_data['password2']
            if password==password2:
                acc = Account(name=name, register_no=regno, sastra_email=sastra_email, branch=branch, batch=batch)
                acc.save()
                msg='Account created Successfully :)'
            else:
                msg = 'Passwords do not match, Please Re-enter them.'
                return render(request,'account_signup.html',{'msg':msg,'form':acc_form})
            return redirect(login(request))
    return render(request,'account_signup.html',{'msg':msg,'form':acc_form})

def login(request):
    msg = ''
    loginform = LoginForm()
    if request.method == 'POST':
        loginform = LoginForm(request.POST)
        if loginform.is_valid():
            regno = loginform.cleaned_data['register_no']
            password = loginform.cleaned_data['password']
            acc = Account.objects.get(register_no=regno)
            mem = Member.objects.get(register_no=regno)
            if mem:
                if mem.password == password:
                    redirect(member_home(request, mem))
                else:
                    msg = 'Wrong password entered :('
                    return render(request, 'login.html', {'msg': msg, 'form': loginform})
            elif acc:
                if acc.password == password:
                    redirect(account_home(request, acc))
                else:
                    msg = 'Wrong password entered :('
                    return render(request, 'login.html', {'msg': msg, 'form': loginform})
            else:
                msg = 'Account Doesnt Exist! Sign up before Logging in !'
                return render(request, 'login.html', {'msg': msg, 'form': loginform})
    return render(request, 'login.html', {'msg': msg, 'form': loginform})

