from django.shortcuts import render
from django.db.models import Sum, Q
from .models import Attendance, Meeting,Member, Contribution
from .forms import MemberForm, AttendanceFileForm, AddMinutes
import csv
import json
import datetime as dt
from django.db.models.functions import Cast

def column(matrix, i):
    return [row[i] for row in matrix]

def subtract(a,b):
    # Create datetime objects for each time (a and b)
    dateTimeA = dt.datetime.combine(dt.date.today(), a)
    dateTimeB = dt.datetime.combine(dt.date.today(), b)
    # Get the difference between datetimes (as timedelta)
    dateTimeDifference = dateTimeA - dateTimeB
    # Divide difference in seconds by number of seconds
    dateTimeDifferenceInHours = dateTimeDifference.total_seconds() 
    return dateTimeDifferenceInHours


# Create your views here. (add_members, view_members, view_meetings(with option to add minutes of meet), upload_attendance_file)

def home(request):
    # Fetch all members and their attendance details

    members = Member.objects.using("tracker").using('tracker').filter(Q(role='Member') | Q(role='Co-ordinator'))
    attendance_data = Attendance.objects.using("tracker").using('tracker').all()
    member_wise_attendance_duration={member.name:0 for member in members}
    for member in members:
        for member_detail in attendance_data:
            if member.name==member_detail.member_name:
                member_wise_attendance_duration[member.name]+=member_detail.duration.total_seconds()
    print(member_wise_attendance_duration)
    # when this loop ends, every members total duration is stored as name:total_duration pair in the dictionary
    # calculate total available meeting hours for each member as well.
    # to do this we need meeting duration details
    meetings=Meeting.objects.using("tracker").using('tracker').all()
    member_wise_total_duration={member.name:0 for member in members}
    for member in members:
        for meeting in meetings:
            if member.joined_on <= meeting.date:
                member_wise_total_duration[member.name]+=subtract(meeting.end_time,meeting.start_time)
    print(member_wise_total_duration)
    if member_wise_total_duration!=0:
    # when this loop ends, every members maximum possible duration is calculated.
    # Now we have to match and calculate percentages and sort
        attendance_percentage={member.name:0 for member in members}
        for i in member_wise_attendance_duration:
            for j in member_wise_total_duration:
                if i==j:
                    print(i)
                    print(member_wise_attendance_duration[i])
                    print(member_wise_total_duration[i])
                    if(member_wise_total_duration[i]!=0):
                        attendance_percentage[i]=member_wise_attendance_duration[i]/member_wise_total_duration[i]*100
                    else:
                        attendance_percentage[i]=0.000001 #dummy init
        print(attendance_percentage)
        #at the end of this loop, attendance_percentage has name, percentage key value pairs
        sorted_attendance_percentage=sorted(attendance_percentage.items(),key=lambda kv: (kv[1],kv[0]))
        print(sorted_attendance_percentage)

        if len(sorted_attendance_percentage)>5:
            return render(request,'tracker-templates/home.html',{'highest_attendees':column(sorted_attendance_percentage[:-7:-1],0),'lowest_attendees':column(sorted_attendance_percentage[0:6],0)})
        else:
            return render(request,'tracker-templates/home.html',{'highest_attendees':column(sorted_attendance_percentage,0)})
    #else:
    #code existed for first run only.
        #return redirect('/upload_attendance_file')

def add_members(request):
    form=MemberForm()
    msg=''
    if request.method=='POST':
        form=MemberForm(request.POST)
        if form.is_valid():
            name=form.cleaned_data['name']
            emailid=form.cleaned_data['emailid']
            regno=form.cleaned_data['regno']
            role=form.cleaned_data['role']
            qs=Member.objects.using("tracker").using('tracker').filter(name=name)
            if len(qs)==0:
                m=Member(name=name,emailid=emailid,regno=regno,role=role,joined_on=dt.date.today())
                m.save()
                msg=f'Member {name} added successfully !'
            else:
                msg="Member with name already exists, use a different name."
        else:
            msg="Form Validation Error"
        return render(request,'tracker-templates/add_member.html',{'form':MemberForm(),'msg':msg})
    return render(request,'tracker-templates/add_member.html',{'form':form,'msg':msg})

def view_members(request):
    qs=Member.objects.using("tracker").using('tracker').all()
    msg=''
    if len(qs)!=0:
        return render(request,'tracker-templates/view_members.html',{'qs':qs,'msg':msg})
    else:
        return render(request,'tracker-templates/view_members.html',{'msg':"No member registered yet"})

def view_meetings(request):
    qs=Meeting.objects.using("tracker").using('tracker').all()
    msg=''
    if len(qs)!=0:
        return render(request,'tracker-templates/view_meetings.html',{'qs':qs,'msg':msg})
    else:
        return render(request,'tracker-templates/view_meetings.html',{'msg':"No meetings added yet !"})
    
def upload_attendance_file(request):
    form=AttendanceFileForm()
    msg=''
    if request.method=='POST':
        form=AttendanceFileForm(request.POST,request.FILES)
        if form.is_valid():
            meeting_code=form.cleaned_data['meeting_code']
            meeting_title=form.cleaned_data['meeting_title']
            meeting_date=form.cleaned_data['meeting_date']
            file=request.FILES['file']
            file_data = file.read().decode('utf-8')
            lines = file_data.split('\n')
            start_time=lines[2].lstrip('"* ')[22:30]
            end_time=lines[3].lstrip('"* ')[20:28]
            attendees=lines[5:]
            #print(attendees)
            for i in attendees:
                if len(i)>0:
                    print(eval(i))
                    i=eval(i)
                    member_name=i[0]
                    first_seen=dt.datetime.strptime(i[1],'%Y-%m-%d %H:%M:%S')
                    duration=dt.datetime.strptime(i[2],'%H:%M:%S')
                    duration=dt.timedelta(hours=duration.hour,minutes=duration.minute,seconds=duration.second)
                    #print(member_name,first_seen,duration)
                    members=Member.objects.using("tracker").using('tracker').filter(name=member_name)
                    if not members:
                        msg=f'Unregistered Person-{member_name} found in meeting, skipped.'
                        continue
                    x=Attendance.objects.using("tracker").using('tracker').filter(member_name=member_name,first_seen=first_seen,duration=duration,meeting_code=meeting_code).first()
                    if x:
                        msg=f'duplicate attendance record for {member_name},{meeting_code} exists and thus was skipped'
                        continue
                    a=Attendance(member_name=member_name,first_seen=first_seen,duration=duration,meeting_code=meeting_code)
                    a.save()
            msg+='Attendance saved successfully from file\n'
            qs=Meeting.objects.using("tracker").using('tracker').filter(code=meeting_code)
            if len(qs)==0:
                m=Meeting(title=meeting_title,date=meeting_date,code=meeting_code,start_time=start_time,end_time=end_time,attendees=attendees)
                m.save()
                msg+='Successfully added meeting\n'
            else:
                msg+='Error creating meeting : Meeting code already exists !'
            return render(request,'tracker-templates/upload_attendance_file.html',{'form':AttendanceFileForm(),'msg':msg})
        else:
            msg='Error in Form Validation'
    return render(request,'tracker-templates/upload_attendance_file.html',{'form':form,'msg':msg})
   
def add_minutes(request,code):
    form=AddMinutes()
    msg=''
    qs=Meeting.objects.using("tracker").using('tracker').filter(code=code)
    if len(qs)==0:
        msg='Invalid meeting code, some error occurred'
    else:
        if request.method=='POST':
            form=AddMinutes(request.POST)
            if form.is_valid():
                minutes=form.cleaned_data['minutes']
                m=Meeting.objects.using("tracker").using('tracker').filter(code=code).first()
                m.minutes_of_meeting=minutes
                m.save()
                msg=f"Successfully added Minutes of meet to {code}"
                return render(request,'ack.html',{'msg':msg})
            else:
                msg='Some error Occurred'
    return render(request,'tracker-templates/add_minutes.html',{'form':form,'msg':msg})
            

def meeting_stats(request):
    meeting_codes = Attendance.objects.using("tracker").using("tracker").values_list("meeting_code", flat=True).distinct()

    if request.method == "POST":
        meeting_code = request.POST.get("meeting_code")
        if meeting_code:
            # Fetch meeting and attendance data for the selected meeting
            meeting = Meeting.objects.using("tracker").using("tracker").filter(code=meeting_code).first()
            attendance_data = Attendance.objects.using("tracker").using("tracker").filter(meeting_code=meeting_code)

            if not meeting or not attendance_data.exists():
                return render(
                    request,
                    "tracker-templates/meeting_stats.html",
                    {"meeting_codes": meeting_codes, "error": "No data found for the selected meeting code."},
                )

            # Calculate the meeting duration
            total_meeting_duration = subtract(meeting.end_time, meeting.start_time)

            # Calculate individual durations and categorize attendees
            under_40_count = 0
            between_40_and_80_count = 0
            above_80_count = 0
            under_40_names = []

            for attendee in attendance_data:
                percentage = (attendee.duration.total_seconds() / total_meeting_duration) * 100
                if percentage < 40:
                    under_40_count += 1
                    under_40_names.append(attendee.member_name)
                elif 40 <= percentage < 80:
                    between_40_and_80_count += 1
                else:
                    above_80_count += 1

            return render(
                request,
                "tracker-templates/meeting_stats.html",
                {
                    "meeting_codes": meeting_codes,
                    "meeting_code": meeting_code,
                    "total_count": under_40_count + between_40_and_80_count + above_80_count,
                    "under_40_count": under_40_count,
                    "between_40_and_80_count": between_40_and_80_count,
                    "above_80_count": above_80_count,
                    "under_40_names": under_40_names,
                },
            )

    return render(request, "tracker-templates/meeting_stats.html", {"meeting_codes": meeting_codes})


def member_stats(request):
    member_names = Member.objects.using("tracker").values_list("name", flat=True).distinct()

    if request.method == "POST":
        member_name = request.POST.get("member_name")
        if not member_name:
            return render(request, "tracker-templates/member_stats.html", {"member_names": member_names, "error": "No member selected."})

        member = Member.objects.using("tracker").filter(name=member_name).first()
        if not member:
            return render(request, "tracker-templates/member_stats.html", {"member_names": member_names, "error": f"Member '{member_name}' not found."})

        join_date = member.joined_on
        meetings = Meeting.objects.using("tracker").filter(date__gte=join_date)
        attendance_data = Attendance.objects.using("tracker").filter(member_name=member_name)

        # Calculate total available meeting hours
        total_meeting_duration = sum(
            subtract(meeting.end_time, meeting.start_time) for meeting in meetings
        )

        # Calculate total attended hours
        total_attended_duration = sum(
            record.duration.total_seconds() for record in attendance_data
        )

        if total_meeting_duration == 0:
            return render(request, "tracker-templates/member_stats.html", {"member_names": member_names, "error": f"No valid meeting data for {member_name}."})

        attendance_percentage = (total_attended_duration / total_meeting_duration) * 100

        # Prepare data for chart
        meeting_durations_json = json.dumps([
            {
                "meeting_code": record["meeting_code"],
                "total_duration": record["total_duration"].total_seconds() if record["total_duration"] else 0
            }
            for record in attendance_data.values("meeting_code").annotate(total_duration=Sum("duration"))
        ])

        return render(
            request,
            "tracker-templates/member_stats.html",
            {
                "member_names": member_names,
                "member_name": member_name,
                "total_attended_duration": total_attended_duration,
                "total_meeting_duration": total_meeting_duration,
                "attendance_percentage": round(attendance_percentage, 2),
                "meeting_durations_json": meeting_durations_json,
            },
        )

    return render(request, "tracker-templates/member_stats.html", {"member_names": member_names})
