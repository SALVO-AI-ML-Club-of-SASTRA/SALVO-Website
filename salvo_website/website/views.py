from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.hashers import check_password
from django.contrib import messages
from django.db import models
from django.http import JsonResponse
from .models import Account, Member, Post, JoinRequest, PostLike
from .forms import AccountRegistrationForm, MemberRegistrationForm, LoginForm, JoinRequestForm
from .tagger import PostTagger
from .tag_dataset import AIdict
import json

from . import safe_parse_tree as spt


tagger = PostTagger(AIdict, title_weight=2.0, max_tags=5, min_score=0.05)

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

            # Check if user exists in either Account or Member table
            if not user and not member:
                messages.error(request, "No account found with this register number. Please check your register number or create a new account.")
            elif user and check_password(password, user.password):
                request.session['user_type'] = 'account'
                request.session['register_no'] = user.register_no
                return redirect(account_dashboard)
            elif member and check_password(password, member.password):
                request.session['user_type'] = 'member'
                request.session['register_no'] = member.register_no
                return redirect(member_dashboard)
            else:
                # User exists but password is wrong
                user_type = "account" if user else "member account"
                messages.error(request, f"Incorrect password for this {user_type}. Please check your password and try again.")
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


def register_member(request):
    if request.session.get('user_type') != 'member':
            return redirect(login)
    if request.method == 'POST':
        form = MemberRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Member registered successfully!")
            return redirect(member_dashboard)
    
        
    register_no = request.session.get('register_no')
    print("Register No:", register_no)
    
    member = Member.objects.filter(register_no=register_no).first()
    print("Member:", member)
    form = MemberRegistrationForm()
    return render(request, 'register_member.html', {'form': form, 'member': member})


def account_dashboard(request):
    if request.session.get('user_type') != 'account':
        return redirect(login)

    register_no = request.session.get('register_no')
    account = Account.objects.get(register_no=register_no)
    posts = Post.objects.all().order_by('-date')
    post_data = []

    all_names = list(Member.objects.values_list('name', flat=True)) + list(
        Account.objects.values_list('name', flat=True))
    all_names = sorted(all_names)  # for binary search

    liked_post_ids = PostLike.objects.filter(register_no=register_no).values_list('post_id', flat=True)

    members_dict = {m.register_no: m.name for m in Member.objects.all()}
    accounts_dict = {a.register_no: a.name for a in Account.objects.all()}

    # For resolving name → reg_no in JS search
    search_member_map = {m.name: m.register_no for m in Member.objects.all()}
    search_account_map = {a.name: a.register_no for a in Account.objects.all()}

    for post in posts:
        reg = post.author_reg_no
        author_name = members_dict.get(reg) or accounts_dict.get(reg) or "Unknown"
        post_data.append((post, author_name))
    return render(request, 'account_dashboard.html', {
        'account': account,
        'posts_with_authors': post_data,
        'liked_post_ids': liked_post_ids,
        'members_dict': members_dict,
        'accounts_dict': accounts_dict,
        'search_names': all_names,
        'search_member_map': search_member_map,
        'search_account_map': search_account_map,
    })


def member_dashboard(request):
    if 'register_no' not in request.session or request.session['user_type'] != 'member':
        return redirect('login')

    register_no = request.session.get('register_no')
    member = Member.objects.get(register_no=register_no)
    posts = Post.objects.all().order_by('-date')

    liked_post_ids = PostLike.objects.filter(register_no=register_no).values_list('post_id', flat=True)

    members_dict = {
        m.register_no: {'name': m.name, 'role': m.club_role} for m in Member.objects.all()
    }
    accounts_dict = {
        a.register_no: {'name': a.name} for a in Account.objects.all()
    }

    # Inside both account_dashboard and member_dashboard views
    all_names = list(Member.objects.values_list('name', flat=True)) + list(
        Account.objects.values_list('name', flat=True))
    all_names = sorted(all_names)  # for binary search

    post_data = []
    for post in posts:
        reg = post.author_reg_no
        if reg in members_dict:
            author_info = members_dict[reg]
            user_type = 'member'
        elif reg in accounts_dict:
            author_info = accounts_dict[reg]
            user_type = 'account'
        else:
            author_info = {'name': 'Unknown'}
            user_type = 'unknown'

        post_data.append((post, author_info, user_type))

    applications = JoinRequest.objects.annotate(
        upvote_count=models.Count('upvotes')
    ).order_by('-upvote_count')

    # For resolving name → reg_no in JS search
    search_member_map = {m.name: m.register_no for m in Member.objects.all()}
    search_account_map = {a.name: a.register_no for a in Account.objects.all()}

    return render(request, 'member_dashboard.html', {
        'member': member,
        'posts_with_authors': post_data,
        'applications': applications,
        'liked_post_ids': liked_post_ids,
        'search_names': all_names,
        'search_member_map': search_member_map,
        'search_account_map': search_account_map,
    })


def create_post(request):
    if request.method == 'POST':
        if 'final_submission' in request.POST:
            # Final step: save the post with tags
            title = request.POST['title']
            content = request.POST['content']
            tags = request.POST.getlist('tags[]')
            print("SOME TAGS",tags)
            reg_no = request.session.get('register_no')
            member = Member.objects.filter(register_no=reg_no).first()
            post = Post.objects.create(
                title=title,
                content=content,
                author_reg_no=reg_no,
                verified=True if member else False,
                tags=json.dumps(tags)
            )
            return redirect('/account_home/' if request.session['user_type'] == 'account' else '/member_home/')

        else:
            # Step 1: identify tags and show for confirmation
            title = request.POST['title']
            content = request.POST['content']
            out_tags = tagger.tag_post(title, content)
            print(out_tags)
            check_dictionary=spt.safety_check(title+"\n"+content, ENGLISH_RATIO_THRESHOLD =0.75, AI_LABEL_THRESHOLD=0.55)
            print(check_dictionary)
            # english_status, ai_label_ratio, nsfw_status
            english_status=check_dictionary['check_english']['status'] # fails implies more than 30% is non-english and less than 50% non ai("PASS" or "FAIL")
            ai_label_ratio=check_dictionary['check_english']['ai_label_ratio'] # implies ai label relevance is bad or good (0 to 100 percentage)
            non_english_words=check_dictionary['check_english']['non_english_words']
            recommendations=[]
            # English check
            if english_status != "PASS":
                recommendations.append(f"⚠️❌ The post content contains too much non-English text such as {non_english_words[:4]} making it harder to evaluate properly.")
        
                # AI label relevance
                if ai_label_ratio < 25:  # threshold can be tuned
                    recommendations.append(f"⚠️❌ The post content may not be relevant (AI relevance is low).")
            else:
                nsfw_status=check_dictionary['check_direct_nsfw']['status'] # fails implies contains explicit words/leet-speak nsfw words ("SFW" or "NSFW/Vulgar")
                regex_offensive=set(check_dictionary['check_direct_nsfw']['nsfw_match_words']\
                            +check_dictionary['check_direct_nsfw']['nsfw_match_words_on_clean']\
                            +check_dictionary['check_direct_nsfw']['nsfw_match_words_on_ultra_clean']) # builds set of explicit words
                
                # Direct NSFW / vulgarity
                if nsfw_status != "SFW":
                    recommendations.append("🚫❌ The post content includes explicit or vulgar words that may not be appropriate.")
                    # Regex detected offensive words
                    if regex_offensive:
                        offensive_list = ", ".join(sorted(regex_offensive))
                        recommendations.append(f"🚫❌ Detected offensive terms: {offensive_list}")
                else:
                    lstm_status=check_dictionary['check_lstm_attention_nsfw']['status'] # reveals lstms opinion (for offensive speech) ("SAFE" OR "UNSAFE")
                    

                    # LSTM opinion
                    if lstm_status != "SAFE":
                        recommendations.append("🚫❌ Our AI model predicts the post content may be unsafe or offensive or unprofessional.")
                        recommendations.append(" Ignore, if no such content is actually present in your post.")
            # Final message
            if not recommendations:
                recommendations = ["✅ Your post content looks safe to upload."]
            else:
                recommendations = recommendations

            return render(request, 'confirm_tags.html', {
                'title': title,
                'content': content,
                'tags': out_tags,
                'available_tags': sorted(list(AIdict.keys())),
                'recommendations': recommendations
            })

    return render(request, 'create_post.html', {'user_type': request.session['user_type']})


def verify_post(request, post_id):
    if request.session.get('user_type') == 'member':
        post = Post.objects.get(post_id=post_id)
        post.verified = True
        post.verified_by = request.session.get('register_no')
        post.save()
    return redirect('member_dashboard')

def delete_post(request, post_id):
    if request.session.get('user_type') != 'member':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': 'Unauthorized'}, status=401)
        return redirect(login)

    #check if memeber is lead or coordinator
    member = Member.objects.get(register_no=request.session.get('register_no'))
    if member.club_role not in ['Lead', 'Co-ordinator']:
        print("Unauthorized delete attempt by:", member.register_no)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': 'You do not have permission to delete posts.'}, status=403)
        messages.error(request, "You do not have permission to delete posts.")
        return redirect('member_dashboard')
    
    try:
        post = Post.objects.get(post_id=post_id)
        print("Post to delete:", post)
        print("Deleting post:", post_id)
        post.delete()
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': 'Post deleted successfully!'
            })
        
        messages.success(request, "Post deleted successfully!")
    except Post.DoesNotExist:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': "Post doesn't exist."}, status=404)
        messages.error(request, "Post doesn't exist.")
    
    print("Redirecting to member dashboard after delete")
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
    
    # Get status filter from request
    status_filter = request.GET.get('status', 'all').lower()
    
    # Base queryset with annotation
    applications_query = JoinRequest.objects.annotate(
        upvote_count=models.Count('upvotes')
    )
    
    # Apply status filter
    if status_filter == 'pending':
        applications = applications_query.filter(status='Pending')
    elif status_filter == 'accepted':
        applications = applications_query.filter(status='Accepted')
    elif status_filter == 'rejected':
        applications = applications_query.filter(status='Rejected')
    else:
        applications = applications_query  # Show all applications
    
    applications = applications.order_by('-upvote_count')

    # Get the applications that the current user has upvoted
    user_upvoted_applications = list(
        JoinRequest.objects.filter(upvotes=member).values_list('id', flat=True)
    )
    
    # Get counts for each status for the filter tabs
    status_counts = {
        'all': JoinRequest.objects.count(),
        'pending': JoinRequest.objects.filter(status='Pending').count(),
        'accepted': JoinRequest.objects.filter(status='Accepted').count(),
        'rejected': JoinRequest.objects.filter(status='Rejected').count(),
    }

    return render(request, 'view_applications.html', {
        'member': member,
        'applications': applications,
        'user_upvoted_applications': user_upvoted_applications,
        'current_filter': status_filter,
        'status_counts': status_counts,
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

    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'new_likes': post.likes,
            'liked': True
        })

    # Regular redirect for non-AJAX requests
    if request.session.get('user_type') == 'member':
        return redirect('member_dashboard')
    else:
        return redirect('account_dashboard')


def account_profile(request, reg_no):
    account = Account.objects.get(register_no=reg_no)
    posts = Post.objects.filter(author_reg_no=reg_no)
    return render(request, 'account_profile.html', {'user': account, 'posts': posts})


def member_profile(request, reg_no):
    member = Member.objects.get(register_no=reg_no)
    posts = Post.objects.filter(author_reg_no=reg_no)
    return render(request, 'member_profile.html', {'user': member, 'posts': posts})

def logout(request):
    # Clear all session data
    request.session.flush()
    messages.success(request, "You have been logged out successfully!")
    return redirect(home)

