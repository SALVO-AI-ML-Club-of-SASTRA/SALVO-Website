from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.hashers import check_password
from django.contrib import messages
from django.db import models
from .models import Account, Member, Post, JoinRequest, PostLike
from .forms import AccountRegistrationForm, MemberRegistrationForm, LoginForm, JoinRequestForm


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
    posts = Post.objects.all()
    post_data = []

    liked_post_ids = PostLike.objects.filter(register_no=register_no).values_list('post_id', flat=True)

    members_dict = {m.register_no: m.name for m in Member.objects.all()}
    accounts_dict = {a.register_no: a.name for a in Account.objects.all()}

    for post in posts:
        reg = post.author_reg_no
        author_name = members_dict.get(reg) or accounts_dict.get(reg) or "Unknown"
        post_data.append((post, author_name))
    return render(request, 'account_dashboard.html', {'account': account,
                                                      'posts_with_authors': post_data,
                                                      'liked_post_ids': liked_post_ids})


def member_dashboard(request):
    if 'register_no' not in request.session or request.session['user_type'] != 'member':
        return redirect('login')

    register_no = request.session.get('register_no')
    member = Member.objects.get(register_no=request.session['register_no'])
    posts = Post.objects.all()

    liked_post_ids = PostLike.objects.filter(register_no=register_no).values_list('post_id', flat=True)

    members_dict = {m.register_no: m.name for m in Member.objects.all()}
    accounts_dict = {a.register_no: a.name for a in Account.objects.all()}

    post_data = []
    for post in posts:
        reg = post.author_reg_no
        author_name = members_dict.get(reg) or accounts_dict.get(reg) or "Unknown"
        post_data.append((post, author_name))

    applications = JoinRequest.objects.annotate(
        upvote_count=models.Count('upvotes')
    ).order_by('-upvote_count')

    return render(request, 'member_dashboard.html', {
        'member': member,
        'posts_with_authors': post_data,
        'applications': applications,
        'liked_post_ids': liked_post_ids
    })


def create_post(request):
    if request.method == 'POST':
        title = request.POST['title']
        content = request.POST['content']
        reg_no = request.session.get('register_no')
        member = Member.objects.filter(register_no=reg_no)
        if member:
            Post.objects.create(title=title, content=content, author_reg_no=reg_no, verified=True)
        else:
            Post.objects.create(title=title, content=content, author_reg_no=reg_no)
        return redirect('/account_home/' if request.session['user_type'] == 'account' else '/member_home/')
    return render(request, 'create_post.html')


def verify_post(request, post_id):
    if request.session.get('user_type') == 'member':
        post = Post.objects.get(post_id=post_id)
        post.verified = True
        post.verified_by = request.session.get('register_no')
        post.save()
    return redirect('member_dashboard')


def join_request(request, reg_no):
    account = Account.objects.get(register_no=reg_no)
    existing = JoinRequest.objects.filter(account=account).first()

    if existing and existing.status == 'Rejected':
        if request.method == 'POST':
            form = JoinRequestForm(request.POST, request.FILES)
            if form.is_valid():
                join_req = form.save(commit=False)
                join_req.account = account
                join_req.status = 'Pending'
                join_req.save()
                return redirect('account_dashboard')
        else:
            form = JoinRequestForm()
        return render(request, 'reapply_join_request.html', {'form': form, 'prev_request': existing})

    elif existing:
        return render(request, 'view_join_request.html', {'join_request': existing})

    else:
        if request.method == 'POST':
            form = JoinRequestForm(request.POST, request.FILES)
            if form.is_valid():
                join_req = form.save(commit=False)
                join_req.account = account
                join_req.save()
                return redirect('account_dashboard')
        else:
            form = JoinRequestForm()
        return render(request, 'join_request.html', {'form': form})


def view_applications(request):
    if 'register_no' not in request.session or request.session['user_type'] != 'member':
        return redirect('login')

    member = Member.objects.get(register_no=request.session['register_no'])
    applications = JoinRequest.objects.annotate(
        upvote_count=models.Count('upvotes')
    ).order_by('-upvote_count')

    return render(request, 'view_applications.html', {
        'member': member,
        'applications': applications,
    })


def upvote_application(request, app_id):
    member = Member.objects.get(register_no=request.session['register_no'])
    app = JoinRequest.objects.get(id=app_id)
    app.upvotes.add(member)
    return redirect('view_applications')


def update_application_status(request, app_id, action):
    member = Member.objects.get(register_no=request.session['register_no'])
    if member.club_role not in ['Lead', 'Coordinator']:
        return HttpResponse("Unauthorized", status=401)

    app = JoinRequest.objects.get(id=app_id)
    if action == 'accept':
        app.status = 'Accepted'
    elif action == 'reject':
        app.status = 'Rejected'
    app.save()
    return redirect('view_applications')


def like_post(request, post_id):
    reg_no = request.session.get('register_no')
    post = Post.objects.get(pk=post_id)

    already_liked = PostLike.objects.filter(post=post, register_no=reg_no).exists()
    if not already_liked:
        PostLike.objects.create(post=post, register_no=reg_no)
        post.likes += 1
        post.save()

    if request.session.get('user_type') == 'member':
        return redirect('member_dashboard')
    else:
        return redirect('account_dashboard')

