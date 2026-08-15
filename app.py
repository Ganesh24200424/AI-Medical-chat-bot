from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_mail import Mail, Message
import random
import string
from werkzeug.utils import secure_filename
import os
import plotly.express as px
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd
import os
import csv
from dotenv import load_dotenv
import pdfkit
from reportlab.pdfgen import canvas
from io import BytesIO
from flask.helpers import send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
mail = Mail(app)
load_dotenv()

app.secret_key = 'MYSECRETKEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = '99220041678@klu.ac.in'
app.config['MAIL_PASSWORD'] = 'soaz uraj onlw ervj'  # Replace this with your App Password
app.config['MAIL_DEFAULT_SENDER'] = ('KARE Healthcare', '99220041678@klu.ac.in')
app.config['ADMIN_EMAIL'] = '99220041678@klu.ac.in'
app.config['MAIL_DEBUG'] = True  # Enable debug mode for email
app.config['MAIL_SUPPRESS_SEND'] = False  # Ensure emails are actually sent

# Initialize Flask-Mail with the app
mail = Mail(app)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
    type_of_doctor = db.Column(db.String(50))
    is_approved = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    blood_group = db.Column(db.String(10), nullable=False)
    time_slot = db.Column(db.String(50), nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    type_of_doctor = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Pending')
    prescription_file = db.Column(db.String(255))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('appointments', lazy=True))

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    report_type = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    notes = db.Column(db.Text)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='Uploaded')

def create_tables():
    with app.app_context():
        db.create_all()

def generate_random_string(length=10):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def send_mail(subject, recipient, body):
    try:
        msg = Message(
            subject=subject,
            recipients=[recipient],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        msg.body = body
        mail.send(msg)
        print(f"Email sent successfully to {recipient}")
        return True
    except Exception as e:
        print(f"Error sending email to {recipient}: {str(e)}")
        # Log the full error details for debugging
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return False
    
# Set the path to the directory containing text files
text_files_dir = os.path.join(os.path.dirname(__file__), 'static/prescriptions')

# Set the path to the directory where PDFs will be saved
pdf_output_dir = os.path.join(os.path.dirname(__file__), 'static/pdfs')

# Function to convert text file to PDF
def convert_to_pdf(file_path, output_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pdfkit.from_string(content, output_path, {'title': 'PDF Conversion', 'footer-center': '[page]/[topage]'})
    


data = pd.read_csv(os.path.join("C:\\Users\\HP\\Downloads\\AHMS (2)\\AHMS\\AHMS\\static\\Data\\Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def predict(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()

    predicted_disease = dt.predict(user_input_label)[0]
    confidence_score = np.max(dt.predict_proba(user_input_label)) * 100  # Assuming decision tree has predict_proba method

    return predicted_disease, confidence_score

with open('C:\\Users\\HP\\Downloads\\AHMS (2)\\AHMS\\AHMS\\static\\Data\\Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]
        

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if not user:
            return redirect(url_for('login'))
            
        username = user.username
        
        if user.type_of_doctor:
            # Get all appointments for this doctor type
            appointments = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor).all()
            
            # Calculate statistics
            total_patients = len(set(appt.name for appt in appointments))
            total_appointments = len(appointments)
            pending_appointments = sum(1 for appt in appointments if appt.status == 'Pending')
            completed_appointments = sum(1 for appt in appointments if appt.status == 'Completed')
            
            # Calculate success rate
            success_rate = (completed_appointments / total_appointments * 100) if total_appointments > 0 else 0
            
            return render_template('doctor-dashboard.html',
                username=username,
                doctor=user,
                total_patients=total_patients,
                total_appointments=total_appointments,
                pending_appointments=pending_appointments,
                success_rate=round(success_rate, 1),
                appointments=appointments
            )
        else:
            user_appointments = user.appointments
            return render_template('patient-dashboard.html', username=username, user_appointments=user_appointments)
            
    return render_template('index.html')


@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    
    # Get statistics for doctor profile
    if user.type_of_doctor:
        total_patients = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor).count()
        total_appointments = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor).count()
        success_rate = 95  # This could be calculated based on completed appointments
        total_consultations = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor).count()
        active_appointments = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor, status='Approved').count()
        satisfaction_rate = 98  # This could be calculated based on patient feedback
        
        return render_template('doctor-profile.html',
                             username=user.username,
                             user=user,
                             total_patients=total_patients,
                             total_appointments=total_appointments,
                             success_rate=success_rate,
                             total_consultations=total_consultations,
                             active_appointments=active_appointments,
                             satisfaction_rate=satisfaction_rate,
                             last_login=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        # Handle patient profile
        user_appointments = user.appointments
        return render_template('patient-profile.html', username=user.username, Email=user.email, user_appointments=user_appointments)

@app.route('/patient-register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            session['user_id'] = user.id
            return redirect(url_for('index'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'error')

    return render_template('patient-register.html')

@app.route('/doctor-register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        type_of_doctor = request.form['type_of_doctor']
        
        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists. Please choose different credentials.', 'error')
            return redirect(url_for('doctor_register'))
        
        # Create new doctor user with pending approval
        user = User(
            username=username,
            email=email,
            password=password,
            type_of_doctor=type_of_doctor,
            is_approved=False
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            
            # Send notification to admin
            admin_subject = 'New Doctor Registration Request - KARE Healthcare'
            admin_body = f'''
            Dear Admin,

            A new doctor has requested registration:

            Doctor Details:
            - Name: {username}
            - Email: {email}
            - Specialization: {type_of_doctor}
            - Registration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Please review and approve this registration at your earliest convenience.

            Best regards,
            KARE Healthcare Team
            '''
            admin_email_sent = send_mail(admin_subject, app.config['ADMIN_EMAIL'], admin_body)
            
            # Send confirmation to doctor
            doctor_subject = 'Registration Request Received - KARE Healthcare'
            doctor_body = f'''
            Dear Dr. {username},

            Thank you for choosing KARE Healthcare for your professional journey.

            Your registration request has been received and is pending admin approval.
            You will receive an email with your login credentials once approved.

            For any queries, please contact our team:
            - G Vasu : vasu@karehealthcare.com
            -  Vijay :vijay@karehealthcare.com
            - chandra: chandra@karehealthcare.com
            - yshwanth sai : sai@karehealthcare.com

            Best regards,
            KARE Healthcare Team
            '''
            doctor_email_sent = send_mail(doctor_subject, email, doctor_body)
            
            if admin_email_sent and doctor_email_sent:
                flash('Registration request submitted. You will receive an email once approved.', 'success')
            else:
                flash('Registration successful, but email notification failed. Please contact admin for approval.', 'warning')
            
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('doctor_register'))
            
    return render_template('doctor-register.html')

@app.route('/admin/approve-doctor/<int:doctor_id>')
def approve_doctor(doctor_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    admin = User.query.get(session['user_id'])
    if not admin or not admin.is_admin:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))
    
    doctor = User.query.get_or_404(doctor_id)
    if doctor.is_approved:
        flash('Doctor is already approved.', 'info')
        return redirect(url_for('admin'))
    
    try:
        # Update doctor status
        doctor.is_approved = True
        db.session.commit()
        
        # Send approval email to doctor
        subject = 'Doctor Registration Approved - KARE Healthcare'
        body = f'''
        Dear Dr. {doctor.username},

        We are pleased to inform you that your registration request has been approved by the admin.
        Welcome to the KARE Healthcare family!

        Your login credentials:
        Username: {doctor.username}
        Password: {doctor.password}

        Please keep your credentials secure and do not share them with anyone.
        You can now log in to your account and start accepting appointments.

        For any assistance, please contact our team:
            - G Vasu : vasu@karehealthcare.com
            -  Vijay :vijay@karehealthcare.com
            - chandra: chandra@karehealthcare.com
            - yshwanth sai : sai@karehealthcare.com

        Best regards,
        KARE Healthcare Management
        '''
        
        email_sent = send_mail(subject, doctor.email, body)
        
        if email_sent:
            flash('Doctor approved successfully and notification email sent.', 'success')
        else:
            flash('Doctor approved successfully, but failed to send notification email. Please contact the doctor directly.', 'warning')
            print(f"Failed to send approval email to {doctor.email}")
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error approving doctor: {str(e)}', 'error')
        print(f"Error in approve_doctor: {str(e)}")
    
    return redirect(url_for('admin'))

@app.route('/change-password', methods=['GET', 'POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        if user.password != current_password:
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('change_password'))
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return redirect(url_for('change_password'))
        
        user.password = new_password
        db.session.commit()
        
        flash('Password changed successfully.', 'success')
        return redirect(url_for('index'))
    
    return render_template('change-password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            if user.type_of_doctor and not user.is_approved:
                flash('Your doctor registration is pending admin approval. Please wait for the approval email.', 'warning')
                return redirect(url_for('login'))
            
            session['user_id'] = user.id
            
            # Redirect admin users to admin dashboard
            if user.is_admin:
                return redirect(url_for('admin'))
            
            return redirect(url_for('index'))
        else:
            flash('Wrong username or password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/book-appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Fetch user details safely
    user = User.query.get(session['user_id'])
    if not user:
        flash("User not found. Please log in again.", "error")
        return redirect(url_for('login'))
    
    username = user.username

    # Fetch distinct types of doctors
    doctor_types = [doctor[0] for doctor in db.session.query(User.type_of_doctor).distinct().all()]

    if request.method == 'POST':
        try:
            name = request.form.get('name')
            age = request.form.get('age', type=int)
            blood_group = request.form.get('blood_group')
            time_slot = request.form.get('time_slot')
            phone_number = request.form.get('phone_number')
            email = request.form.get('email')
            type_of_doctor = request.form.get('type_of_doctor')

            if not all([name, age, blood_group, time_slot, phone_number, email, type_of_doctor]):
                flash("All fields are required.", "error")
                return redirect(url_for('book_appointment'))

            appointment = Appointment(
                name=name,
                age=age,
                blood_group=blood_group,
                time_slot=time_slot,
                phone_number=phone_number,
                email=email,
                type_of_doctor=type_of_doctor,
                user=user
            )

            db.session.add(appointment)
            db.session.commit()

            # Get doctor email, use a default email if no doctor found
            doctor = User.query.filter_by(type_of_doctor=type_of_doctor).first()
            doctor_email = doctor.email if doctor and doctor.email else "99220041678@klu.ac"

            subject = 'New Appointment Request'
            body = f'Hello Doctor,\n\nYou have a new appointment request from {name}. Please log in to approve or reject it.'

            send_mail(subject, doctor_email, body)

            flash("Appointment booked successfully!", "success")
            return redirect(url_for('index'))
        
        except Exception as e:
            db.session.rollback()  # Rollback in case of database failure
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('book_appointment'))

    return render_template('book-appointment.html', doctor_types=doctor_types, username=username)



@app.route('/approve-appointment/<int:appointment_id>')
def approve_appointment(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    appointment = Appointment.query.get_or_404(appointment_id)
    user = User.query.get(session['user_id'])
    
    if not user.type_of_doctor or user.type_of_doctor != appointment.type_of_doctor:
        flash('You are not authorized to approve this appointment.', 'error')
        return redirect(url_for('index'))
    
    try:
        appointment.status = 'Approved'
        db.session.commit()
        
        # Send email notification to patient with video call details
        subject = 'Appointment Approved - Video Consultation Scheduled'
        body = f'''
        Dear {appointment.name},

        Your appointment with Dr. {user.username} has been approved for {appointment.time_slot}.

        Video Consultation Details:
        - Date & Time: {appointment.time_slot}
        - Doctor: Dr. {user.username}
        - Specialization: {user.type_of_doctor}

        To join the video consultation:
        1. Log in to your KARE Healthcare account
        2. Go to "My Appointments"
        3. Click on "Join Video Call" at the scheduled time

        For any assistance, please contact our support team:
        - Email: support@karehealthcare.com
        - Phone: +91 9876543210

        Best regards,
        KARE Healthcare Team
        '''
        
        email_sent = send_mail(subject, appointment.email, body)
        
        if email_sent:
            flash('Appointment approved successfully and video consultation details sent to patient.', 'success')
        else:
            flash('Appointment approved, but failed to send email notification. Please contact the patient directly.', 'warning')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error approving appointment: {str(e)}', 'error')
        print(f"Error in approve_appointment: {str(e)}")
    
    return redirect(url_for('doctor_patients'))

@app.route('/reject-appointment/<int:appointment_id>')
def reject_appointment(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    appointment = Appointment.query.get_or_404(appointment_id)
    user = User.query.get(session['user_id'])
    
    if not user.type_of_doctor or user.type_of_doctor != appointment.type_of_doctor:
        flash('You are not authorized to reject this appointment.', 'error')
        return redirect(url_for('index'))
    
    appointment.status = 'Rejected'
    db.session.commit()
    
    # Send email notification to patient
    subject = 'Appointment Rejected'
    body = f'Dear {appointment.name},\n\nWe regret to inform you that your appointment with Dr. {user.username} for {appointment.time_slot} has been rejected.\n\nPlease try booking another time slot.\n\nBest regards,\nKARE Team'
    send_mail(subject, appointment.email, body)
    
    flash('Appointment rejected successfully!', 'success')
    return redirect(url_for('doctor_patients'))

@app.route('/policy')
def policy():
    return render_template('privacy-policy.html')

@app.route('/Transforming_Healthcare')
def Transforming_Healthcare():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
    return render_template('index.html')

@app.route('/Holistic_Health')
def Holistic_Health():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
    return render_template('index.html')

@app.route('/Nourishing_Body')
def Nourishing_Body():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
    return render_template('index.html')

@app.route('/Importance_of_Games')
def Importance_of_Games():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
    return render_template('index.html')

@app.route('/admin')
def admin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    admin_user = User.query.get(session['user_id'])
    if not admin_user or not admin_user.is_admin:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get statistics
        total_doctors = User.query.filter(User.type_of_doctor.isnot(None), User.is_approved==True).count()
        pending_doctors = User.query.filter(User.type_of_doctor.isnot(None), User.is_approved==False).count()
        total_patients = User.query.filter(User.type_of_doctor.is_(None), User.is_admin==False).count()
        total_appointments = Appointment.query.count()
        
        # Get pending doctors with proper ordering
        pending_doctors_list = User.query.filter(
            User.type_of_doctor.isnot(None),
            User.is_approved==False
        ).order_by(User.registration_date.desc()).all()
        
        # Get recent appointments with proper ordering
        recent_appointments = Appointment.query.order_by(
            Appointment.time_slot.desc()
        ).limit(10).all()
        
        # Get doctor-wise patient counts
        doctor_stats = []
        approved_doctors = User.query.filter(User.type_of_doctor.isnot(None), User.is_approved==True).all()
        for doctor in approved_doctors:
            patient_count = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).count()
            doctor_stats.append({
                'name': doctor.username,
                'type': doctor.type_of_doctor,
                'patient_count': patient_count
            })
        
        return render_template('admin.html',
                             username=admin_user.username,
                             is_admin=True,
                             total_doctors=total_doctors,
                             pending_doctors=pending_doctors,
                             total_patients=total_patients,
                             total_appointments=total_appointments,
                             pending_doctors_list=pending_doctors_list,
                             recent_appointments=recent_appointments,
                             doctor_stats=doctor_stats)
                             
    except Exception as e:
        flash(f'Error loading admin dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/videocall')
def videocall():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    appointment_id = request.args.get('appointment')
    if appointment_id:
        appointment = Appointment.query.get_or_404(appointment_id)
        if appointment.status != 'Approved':
            flash('This appointment is not approved for video consultation.', 'error')
            return redirect(url_for('doctor_patients'))
    
    user = User.query.get(session['user_id'])
    # return render_template('videocall.html', username=user.username, appointment=appointment if appointment_id else None)
    return redirect("https://webrtc-3-vtsp.onrender.com/")

@app.route('/doctor-patients')
def doctor_patients():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username

        doctor = User.query.get(session['user_id'])

        if not doctor.type_of_doctor:
            return redirect(url_for('index'))

        # Fetch appointments assigned to the doctor
        appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).all()
        file_list = os.listdir(text_files_dir)

        return render_template('doctor-patients.html', doctor=doctor, appointments=appointments,username=username,file_list=file_list)
    return render_template('index.html')

@app.route('/prescribe-medicine/<int:appointment_id>', methods=['GET', 'POST'])
def prescribe_medicine(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor:
        return redirect(url_for('index'))

    available_medicines = ["Medicine 1", "Medicine 2", "Medicine 3"]  # Update this with your list of medicines

    if request.method == 'POST':
        selected_medicines = request.form.getlist('medicines[]')

        # Create a PDF document using ReportLab
        buffer = BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=letter)

        # Define styles for the header and footer
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle(
            'Header1',
            parent=styles['Heading1'],
            fontName='monospace',
            fontSize=18,
            spaceAfter=12,
            textColor=colors.green,
        )

        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
        )

        # Create content for the PDF
        content = []

        # Add Jansevak header with green color
        jansevak_header = Paragraph("<font color='green' size='24'><b>KARE Healthcare: We Care for Your Health</b></font>", header_style)
        content.append(jansevak_header)

        # Add space after Jansevak header
        content.append(Spacer(1, 12))

        # Add patient details
        patient_details = (
            f"<b>Patient Details:</b><br/>"
            f"Name: {appointment.name}<br/>"
            f"Age: {appointment.age}<br/>"
            f"Blood Group: {appointment.blood_group}<br/>"
            f"Phone Number: {appointment.phone_number}"
        )
        content.append(Paragraph(patient_details, styles['Normal']))

        # Add space after patient details
        content.append(Spacer(1, 12))

        # Add prescribed medicines
        prescribed_meds = "<b>Prescribed Medicines:</b><br/>"
        for medicine in selected_medicines:
            prescribed_meds += f"- {medicine}<br/>"
        content.append(Paragraph(prescribed_meds, styles['Normal']))

        # Add space after prescribed medicines
        content.append(Spacer(1, 12))

        # Add doctor details and footer
        doctor_details = (
            f"<b>Prescribed by Dr. {doctor.username} ({doctor.type_of_doctor})</b><br/>"
            "Thank you for choosing Kare! We wish you good health."
        )
        content.append(Paragraph(doctor_details, styles['Normal']))

        # Add space after doctor details
        content.append(Spacer(1, 12))

        # Build the PDF
        pdf.build(content)

        # Save the PDF to the file
        pdf_filename = f"prescription_{appointment_id}.pdf"
        pdf_filepath = os.path.join("static", "prescriptions", pdf_filename)
        buffer.seek(0)
        with open(pdf_filepath, 'wb') as pdf_file:
            pdf_file.write(buffer.read())

        buffer.close()

        # Update appointment status to 'Prescribed'
        appointment.status = 'Prescribed'
        appointment.prescription_file = pdf_filepath
        db.session.commit()

        return redirect(url_for('doctor_patients'))

    return render_template('prescribe-medicine.html', appointment=appointment, available_medicines=available_medicines)
    
@app.route('/view-prescription/<int:appointment_id>')
def view_prescription(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor or appointment.status != 'Prescribed':
        return redirect(url_for('index'))

    prescription_filepath = appointment.prescription_file

    return send_file(prescription_filepath, as_attachment=True)

@app.route('/view-prescription-patient/<int:appointment_id>')
def view_prescription_patient(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if not user or not appointment or appointment.user_id != user.id or appointment.status != 'Prescribed':
        return redirect(url_for('profile'))  # Change this line to redirect to the patient's profile instead of index

    # Read prescription text from the file
    prescription_filepath = appointment.prescription_file
    

    return send_file(prescription_filepath, as_attachment=True)

# ============================================================ scans ============================================================ 
    
@app.route('/braintumor', methods=['GET', 'POST'])
def braintumor():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('brain-tumor.html',username=username)
    else:
        return render_template('index.html')
    

@app.route('/disease_predict', methods=['GET', 'POST'])
def disease_predict():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        chart_data={}
        if request.method == 'POST':
            selected_symptoms = []
            if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom1'])
            if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom2'])
            if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom3'])
            if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom4'])
            if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom5'])
            disease, confidence_score = predict(selected_symptoms)
            
            chart_data = {
            'disease': disease,
            'confidence_score': confidence_score
            }
            return render_template('disease_predict.html',symptoms=symptoms,disease=disease, chart_data=chart_data,confidence_score=confidence_score,username=username)
            
        return render_template('disease_predict.html',symptoms=symptoms,username=username,chart_data=chart_data)
    else:
        return render_template('index.html')

@app.route('/lung')
def lung():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('lung.html',username=username)
    else:
        return render_template('index.html')

@app.route('/cataract')
def cataract():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('cataract.html',username=username)
    return render_template('index.html')

def verify_admin_user():
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@kare.com',
                password='admin123',
                is_admin=True,
                is_approved=True,
                type_of_doctor=None
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully!")
        else:
            # Update admin user if it exists
            admin.is_admin = True
            admin.is_approved = True
            admin.password = 'admin123'  # Reset password to default
            db.session.commit()
            print("Admin user updated successfully!")

@app.route('/admin/reject-doctor/<int:doctor_id>')
def reject_doctor(doctor_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    admin = User.query.get(session['user_id'])
    if not admin or not admin.is_admin:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))
    
    doctor = User.query.get_or_404(doctor_id)
    if doctor.is_approved:
        flash('Doctor is already approved.', 'info')
        return redirect(url_for('admin'))
    
    # Send rejection email to doctor
    subject = 'Doctor Registration Rejected'
    body = f'''
    Dear {doctor.username},
    
    We regret to inform you that your registration request has been rejected by the admin.
    
    If you believe this is a mistake, please contact the admin at {app.config['ADMIN_EMAIL']}
    
    Best regards,
    KARE Management
    '''
    send_mail(subject, doctor.email, body)
    
    # Delete the doctor's account
    db.session.delete(doctor)
    db.session.commit()
    
    flash('Doctor registration rejected successfully.', 'success')
    return redirect(url_for('admin'))

def init_db():
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

@app.route('/test-email')
def test_email():
    try:
        subject = "Test Email from KARE Healthcare"
        recipient = app.config['MAIL_USERNAME']  # Send to yourself for testing
        body = "This is a test email to verify email functionality."
        
        success = send_mail(subject, recipient, body)
        if success:
            return "Test email sent successfully!"
        else:
            return "Failed to send test email. Check server logs for details."
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/doctor-dashboard')
def doctor_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if session.get('user_type') != 'doctor':
        return redirect(url_for('login'))
    
    try:
        # Get doctor details
        doctor = User.query.filter_by(username=session['username']).first()
        if not doctor:
            return redirect(url_for('login'))

        # Get statistics
        total_patients = len(set(appt.name for appt in Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).all()))
        total_appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).count()
        pending_appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor, status='Pending').count()
        completed_appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor, status='Completed').count()
        
        # Calculate success rate
        success_rate = (completed_appointments / total_appointments * 100) if total_appointments > 0 else 0
        
        # Get recent appointments
        appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor)\
            .order_by(Appointment.time_slot.desc())\
            .limit(10)\
            .all()
        
        # Get patients list
        patients = []
        for appt in Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).all():
            patient = {
                'id': appt.id,
                'name': appt.name,
                'age': appt.age,
                'last_visit': appt.time_slot,
                'next_appointment': None,
                'status': appt.status
            }
            # Find next appointment if any
            next_appt = Appointment.query.filter_by(
                type_of_doctor=doctor.type_of_doctor,
                name=appt.name,
                status='Approved'
            ).filter(Appointment.time_slot > datetime.now())\
            .order_by(Appointment.time_slot.asc())\
            .first()
            
            if next_appt:
                patient['next_appointment'] = next_appt.time_slot
            
            if patient not in patients:
                patients.append(patient)

        # Get reports
        reports = []
        # Add report fetching logic here if you have a Reports table

        return render_template('doctor-dashboard.html',
            username=session['username'],
            doctor=doctor,
            total_patients=total_patients,
            total_appointments=total_appointments,
            pending_appointments=pending_appointments,
            success_rate=round(success_rate, 1),
            appointments=appointments,
            patients=patients,
            reports=reports
        )
    except Exception as e:
        print(f"Error in doctor_dashboard: {str(e)}")
        return redirect(url_for('login'))

@app.route('/upload-report', methods=['POST'])
def upload_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    doctor = User.query.get(session['user_id'])
    if not doctor or not doctor.type_of_doctor:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))
    
    try:
        patient_id = request.form.get('patient_id')
        report_type = request.form.get('report_type')
        notes = request.form.get('notes')
        report_file = request.files.get('report_file')
        
        if not all([patient_id, report_type, report_file]):
            flash('Please fill all required fields.', 'error')
            return redirect(url_for('doctor_dashboard'))
        
        # Save the file
        filename = secure_filename(f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_file.filename}")
        file_path = os.path.join('static', 'reports', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        report_file.save(file_path)
        
        # Create report record
        report = Report(
            patient_id=patient_id,
            doctor_id=doctor.id,
            report_type=report_type,
            file_path=file_path,
            notes=notes
        )
        
        db.session.add(report)
        db.session.commit()
        
        flash('Report uploaded successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error uploading report: {str(e)}', 'error')
        print(f"Error in upload_report: {str(e)}")
    
    return redirect(url_for('doctor_dashboard'))

@app.route('/view-report/<int:report_id>')
def view_report(report_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    report = Report.query.get_or_404(report_id)
    user = User.query.get(session['user_id'])
    
    # Check if user has permission to view the report
    if not user.is_admin and user.id != report.doctor_id and user.id != report.patient_id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))
    
    return send_file(report.file_path)

@app.route('/download-report/<int:report_id>')
def download_report(report_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    report = Report.query.get_or_404(report_id)
    user = User.query.get(session['user_id'])
    
    # Check if user has permission to download the report
    if not user.is_admin and user.id != report.doctor_id and user.id != report.patient_id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))
    
    return send_file(report.file_path, as_attachment=True)

if __name__ == '__main__':
    init_db()
    verify_admin_user()
    app.run(debug=True)