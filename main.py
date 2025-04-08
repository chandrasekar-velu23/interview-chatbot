import os
import random
import datetime
import json
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from flask import Flask, jsonify, request, render_template, session, send_from_directory
from flask.sessions import SecureCookieSessionInterface
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"NLTK download error: {e}")

# Initialize NLTK components
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Helper function to convert numpy types in any Python object
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Function to generate smart, concise responses based on user input
def generate_concise_response(user_message, context, step):
    """Generate a concise response based on user message and context"""
    # Analyze sentiment
    sentiment = sia.polarity_scores(user_message)
    sentiment_compound = sentiment['compound']
    
    # Tokenize and get keywords
    tokens = word_tokenize(user_message.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    
    # Simple acknowledgment phrases based on sentiment
    if sentiment_compound > 0.3:
        response = "Great!"
    elif sentiment_compound > 0:
        response = "Thanks for sharing."
    elif sentiment_compound > -0.3:
        response = "I understand."
    else:
        response = "I appreciate your honesty."
    
    # Add context-specific brief responses
    context_responses = {
        'general': "Let's continue.",
        'occupation': f"Good to know.",
        'job_role': f"Excellent choice.",
        'job_type': "Perfect.",
        'job_mode': "Got it.",
        'resume_upload': "Almost done!",
        'complete': "Thanks for completing the interview!"
    }
    
    # Look for keywords to personalize slightly
    if any(word in tokens for word in ['experience', 'worked', 'job', 'project']):
        response = "Your experience is valuable."
    elif any(word in tokens for word in ['learn', 'education', 'study']):
        response = "Great background."
    
    # Add the context-specific part
    if step in context_responses:
        response = f"{response} {context_responses[step]}"
    
    return response

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.wsgi_app = ProxyFix(app.wsgi_app)

# Ensure required directories exist
os.makedirs('data', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to track session state
QUESTIONS_FILE = 'questions.csv'
RESULTS_FILE = 'interview_results.xlsx'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_id():
    """Generate a sequential numeric ID based on existing data"""
    try:
        if os.path.exists(RESULTS_FILE):
            df = pd.read_excel(RESULTS_FILE)
            if not df.empty and 'interview_id' in df.columns:
                # Get the highest ID and increment by 1
                max_id = df['interview_id'].max()
                # Convert numpy int64 to native Python int
                if isinstance(max_id, (np.integer, np.int64)):
                    max_id = int(max_id)
                return max_id + 1
            return 1
        return 1
    except Exception as e:
        print(f"Error generating ID: {e}")
        return 1

def get_general_questions():
    """Get the initial general questions from the CSV file"""
    try:
        df = pd.read_csv(QUESTIONS_FILE, encoding='ISO-8859-1')
        questions = []
        for i in range(len(df)):
            question = df.iloc[i, 0]
            if pd.isna(question) or question.strip() == '':
                break
            questions.append(question)
        return questions
    except Exception as e:
        print(f"Error loading general questions: {e}")
        return ["Sorry, I couldn't load the questions. Please try again later."]

def get_occupation_questions(occupation):
    """Get occupation-specific questions"""
    try:
        df = pd.read_csv(QUESTIONS_FILE, encoding='ISO-8859-1')
        occupation_columns = {
            'Student': 'Student', 
            'Fresher': 'Fresher', 
            'Experienced Professional': 'Experienced'
        }
        
        if occupation in occupation_columns:
            column = occupation_columns[occupation]
            questions = df[column].dropna().tolist()
            return random.sample(questions, min(5, len(questions)))
        return []
    except Exception as e:
        print(f"Error loading occupation questions: {e}")
        return []

def get_job_role_questions(role):
    """Get job role specific questions"""
    try:
        df = pd.read_csv(QUESTIONS_FILE, encoding='ISO-8859-1')
        job_roles = {"UI/UX": "UI/UX", "Java": "Java", "AI/ML": "AI/ML"}
        
        if role in job_roles:
            column = job_roles[role]
            if column in df.columns:
                questions = df[column].dropna().tolist()
                return random.sample(questions, min(5, len(questions)))
        return []
    except Exception as e:
        print(f"Error loading job role questions: {e}")
        return []

def save_interview_data(data):
    """Save interview data to Excel file"""
    try:
        # Convert any numpy types to Python native types
        data = convert_numpy_types(data)
                
        # Create DataFrame from the data
        df = pd.DataFrame([data])
        
        # Check if file exists to append or create new
        if os.path.exists(RESULTS_FILE):
            existing_df = pd.read_excel(RESULTS_FILE)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(RESULTS_FILE, index=False)
        else:
            df.to_excel(RESULTS_FILE, index=False)
            
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

@app.route('/')
def index():
    # Initialize a new session
    session['chat_history'] = []
    session['current_step'] = 'general'
    session['question_index'] = 0
    session['responses'] = {}
    session['questions'] = {}  # Store questions along with answers
    session['interview_start_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    session['interview_id'] = int(generate_unique_id())  # Convert to Python int explicitly
    session['resume_uploaded'] = False
    
    return render_template('index.html')

@app.route('/get_message', methods=['POST'])
def get_message():
    try:
        user_message = request.form['user_message'].strip()
        
        # Initialize response variables
        bot_response = ""
        next_question = ""
        is_options = False
        options = []
        is_file_upload = False
        
        # Get current state
        current_step = session.get('current_step', 'general')
        question_index = session.get('question_index', 0)
        
        # Store user's response to the previous question
        if current_step != 'complete' and 'current_question' in session:
            question_key = f"{current_step}_{question_index-1}" if question_index > 0 else "start"
            session['responses'][question_key] = user_message
            # Store the question text as well
            if 'current_question' in session:
                session['questions'][question_key] = session['current_question']
        
        # Process based on current step
        if current_step == 'general':
            general_questions = get_general_questions()
            
            if question_index < len(general_questions):
                # Generate concise response
                if user_message:
                    bot_response = generate_concise_response(user_message, session, current_step)
                else:
                    bot_response = "Welcome! Let's begin."
                
                next_question = general_questions[question_index]
                session['current_question'] = next_question
                session['question_index'] = question_index + 1
            else:
                # Move to occupation questions
                bot_response = generate_concise_response(user_message, session, current_step)
                next_question = "What is your current occupation status?"
                is_options = True
                options = ["Student", "Fresher", "Experienced Professional"]
                session['current_step'] = 'occupation'
                session['current_question'] = next_question
                session['question_index'] = 0
        
        elif current_step == 'occupation':
            if question_index == 0:
                # Process occupation selection
                if user_message in ["1", "2", "3"] or user_message in ["Student", "Fresher", "Experienced Professional"]:
                    occupation_dict = {'1': 'Student', '2': 'Fresher', '3': 'Experienced Professional'}
                    selected_occupation = user_message if user_message in occupation_dict.values() else occupation_dict[user_message]
                    
                    session['selected_occupation'] = selected_occupation
                    session['responses']['occupation'] = selected_occupation
                    session['questions']['occupation'] = "What is your current occupation status?"
                    
                    # Get occupation-specific questions
                    session['occupation_questions'] = get_occupation_questions(selected_occupation)
                    
                    if session['occupation_questions']:
                        # Brief responses based on occupation
                        occupation_responses = {
                            'Student': "Great! Student perspective is valuable.",
                            'Fresher': "Fresh talent is always welcome!",
                            'Experienced Professional': "Your experience matters."
                        }
                        bot_response = occupation_responses.get(selected_occupation, f"{selected_occupation} selected.")
                        next_question = session['occupation_questions'][0]
                        session['current_question'] = next_question
                        session['question_index'] = 1
                    else:
                        # No occupation questions, move to job roles
                        bot_response = f"{selected_occupation} noted. Let's talk about your interests."
                        next_question = "Select a job role:"
                        is_options = True
                        options = ["UI/UX", "Java", "AI/ML"]
                        session['current_step'] = 'job_role'
                        session['current_question'] = next_question
                        session['question_index'] = 0
                else:
                    # Invalid input
                    bot_response = "Please select a valid option."
                    next_question = "What is your current occupation status?"
                    is_options = True
                    options = ["Student", "Fresher", "Experienced Professional"]
                    session['current_question'] = next_question
            
            else:
                # Process occupation question responses
                occupation_questions = session.get('occupation_questions', [])
                
                if question_index <= len(occupation_questions):
                    question_key = f"occupation_q{question_index}"
                    session['responses'][question_key] = user_message
                    session['questions'][question_key] = session['current_question']
                    
                    if question_index < len(occupation_questions):
                        # More occupation questions
                        bot_response = generate_concise_response(user_message, session, current_step)
                        next_question = occupation_questions[question_index]
                        session['current_question'] = next_question
                        session['question_index'] = question_index + 1
                    else:
                        # Move to job roles
                        bot_response = generate_concise_response(user_message, session, current_step)
                        next_question = "Select a job role:"
                        is_options = True
                        options = ["UI/UX", "Java", "AI/ML"]
                        session['current_step'] = 'job_role'
                        session['current_question'] = next_question
                        session['question_index'] = 0
        
        elif current_step == 'job_role':
            if question_index == 0:
                # Process job role selection
                job_roles = {
                    "1": "UI/UX", 
                    "2": "Java", 
                    "3": "AI/ML",
                    "UI/UX": "UI/UX", 
                    "Java": "Java", 
                    "AI/ML": "AI/ML"
                }
                
                if user_message in job_roles:
                    selected_role = job_roles[user_message]
                    session['selected_role'] = selected_role
                    session['responses']['job_role'] = selected_role
                    session['questions']['job_role'] = "Select a job role:"
                    
                    # Get job role questions
                    session['job_role_questions'] = get_job_role_questions(selected_role)
                    
                    # Brief responses based on role
                    role_responses = {
                        "UI/UX": "Great choice! UI/UX is in demand.",
                        "Java": "Java skills are always valuable.",
                        "AI/ML": "AI/ML is cutting-edge. Good pick!"
                    }
                    
                    if session['job_role_questions']:
                        bot_response = role_responses.get(selected_role, f"{selected_role} noted.")
                        next_question = session['job_role_questions'][0]
                        session['current_question'] = next_question
                        session['question_index'] = 1
                    else:
                        # No job role questions, move to job type
                        bot_response = role_responses.get(selected_role, f"{selected_role} noted.")
                        next_question = "Select the type of job role:"
                        is_options = True
                        options = ["Full-time", "Part-time", "Freelancing"]
                        session['current_step'] = 'job_type'
                        session['current_question'] = next_question
                        session['question_index'] = 0
                else:
                    # Invalid input
                    bot_response = "Please select a valid job role."
                    next_question = "Select a job role:"
                    is_options = True
                    options = ["UI/UX", "Java", "AI/ML"]
                    session['current_question'] = next_question
                    
            else:
                # Process job role question responses
                job_role_questions = session.get('job_role_questions', [])
                
                if question_index <= len(job_role_questions):
                    question_key = f"job_role_q{question_index}"
                    session['responses'][question_key] = user_message
                    session['questions'][question_key] = session['current_question']
                    
                    if question_index < len(job_role_questions):
                        # More job role questions
                        bot_response = generate_concise_response(user_message, session, current_step)
                        next_question = job_role_questions[question_index]
                        session['current_question'] = next_question
                        session['question_index'] = question_index + 1
                    else:
                        # Move to job type
                        bot_response = generate_concise_response(user_message, session, current_step)
                        next_question = "Select the type of job role:"
                        is_options = True
                        options = ["Full-time", "Part-time", "Freelancing"]
                        session['current_step'] = 'job_type'
                        session['current_question'] = next_question
                        session['question_index'] = 0
        
        elif current_step == 'job_type':
            # Process job type selection
            job_type_dict = {
                '1': 'Full-time', 
                '2': 'Part-time', 
                '3': 'Freelancing',
                'Full-time': 'Full-time', 
                'Part-time': 'Part-time', 
                'Freelancing': 'Freelancing'
            }
            
            if user_message in job_type_dict:
                selected_job_type = job_type_dict[user_message]
                session['responses']['job_type'] = selected_job_type
                session['questions']['job_type'] = "Select the type of job role:"
                
                # Brief responses based on job type
                type_responses = {
                    "Full-time": "Full-time - solid choice.",
                    "Part-time": "Part-time offers good flexibility.",
                    "Freelancing": "Freelancing gives you independence."
                }
                
                # Move to job mode
                bot_response = type_responses.get(selected_job_type, f"{selected_job_type} noted.")
                next_question = "Select your preferred job mode:"
                is_options = True
                options = ["Remote", "Onsite", "Hybrid"]
                session['current_step'] = 'job_mode'
                session['current_question'] = next_question
            else:
                # Invalid input
                bot_response = "Please select a valid option."
                next_question = "Select the type of job role:"
                is_options = True
                options = ["Full-time", "Part-time", "Freelancing"]
                session['current_question'] = next_question
        
        elif current_step == 'job_mode':
            # Process job mode selection
            job_mode_dict = {
                '1': 'Remote', 
                '2': 'Onsite', 
                '3': 'Hybrid',
                'Remote': 'Remote', 
                'Onsite': 'Onsite', 
                'Hybrid': 'Hybrid'
            }
            
            if user_message in job_mode_dict:
                selected_job_mode = job_mode_dict[user_message]
                session['responses']['job_mode'] = selected_job_mode
                session['questions']['job_mode'] = "Select your preferred job mode:"
                
                # Brief responses based on job mode
                mode_responses = {
                    "Remote": "Remote work - good choice.",
                    "Onsite": "Onsite offers great collaboration.",
                    "Hybrid": "Hybrid gives you flexibility."
                }
                
                # Move to resume upload
                bot_response = mode_responses.get(selected_job_mode, f"{selected_job_mode} selected.")
                next_question = "Please upload your resume (PDF, DOC, or DOCX format)."
                session['current_step'] = 'resume_upload'
                session['current_question'] = next_question
                is_file_upload = True
            else:
                # Invalid input
                bot_response = "Please select a valid option."
                next_question = "Select your preferred job mode:"
                is_options = True
                options = ["Remote", "Onsite", "Hybrid"]
                session['current_question'] = next_question
        
        elif current_step == 'resume_upload':
            if session.get('resume_uploaded', False):
                # Resume already uploaded, complete the interview
                bot_response = "Thanks for completing the interview! Your responses have been recorded."
                next_question = "Would you like to start another interview?"
                is_options = True
                options = ["Yes", "No"]
                session['current_step'] = 'complete'
                
                # Record submission time
                submission_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Prepare data for saving
                data = {
                    'interview_id': session['interview_id'],
                    'interview_start_time': session['interview_start_time'],
                    'submission_time': submission_time,
                    'resume_filename': session.get('resume_filename', 'Not uploaded')
                }
                
                # Add all responses and questions
                for key, value in session['responses'].items():
                    data[f"answer_{key}"] = value
                
                for key, value in session['questions'].items():
                    data[f"question_{key}"] = value
                    
                # Save to Excel
                save_interview_data(data)
            else:
                # Waiting for resume upload
                bot_response = "Please upload your resume to continue."
                next_question = "Please upload your resume (PDF, DOC, or DOCX format)."
                is_file_upload = True
        
        elif current_step == 'complete':
            # Handle restart or end
            if user_message.lower() == 'yes' or user_message == '1':
                # Restart interview
                session['chat_history'] = []
                session['current_step'] = 'general'
                session['question_index'] = 0
                session['responses'] = {}
                session['questions'] = {}
                session['interview_start_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                session['interview_id'] = int(generate_unique_id())  # Convert to Python int explicitly
                session['resume_uploaded'] = False
                
                general_questions = get_general_questions()
                if general_questions:
                    bot_response = "Starting a new interview."
                    next_question = general_questions[0]
                    session['current_question'] = next_question
                    session['question_index'] = 1
                else:
                    bot_response = "Sorry, questions couldn't be loaded. Please try again later."
            else:
                # End the chat
                bot_response = "Thank you for using our interview chatbot. Have a great day!"
                next_question = ""
        
        # Update chat history
        chat_history = session.get('chat_history', [])
        
        if user_message:
            chat_history.append({"sender": "user", "message": user_message})
        
        if bot_response:
            chat_history.append({"sender": "bot", "message": bot_response})
        
        session['chat_history'] = chat_history

        # Ensure all session data is using native Python types
        for key in session:
            session[key] = convert_numpy_types(session[key])
        
        # Return response
        return jsonify({
            "bot_response": bot_response,
            "next_question": next_question,
            "is_options": is_options,
            "options": options,
            "is_file_upload": is_file_upload
        })
    
    except Exception as e:
        print(f"Error in get_message: {e}")
        # Return a friendly error message
        return jsonify({
            "bot_response": "Something went wrong. Let's try again.",
            "next_question": "Please retry.",
            "is_options": False,
            "options": [],
            "is_file_upload": False
        })

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        # Check if the post request has the file part
        if 'resume' not in request.files:
            return jsonify({"success": False, "message": "No file part"})
        
        file = request.files['resume']
        
        # If user does not select a file
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"})
        
        if file and allowed_file(file.filename):
            # Create a unique filename
            filename = f"{session['interview_id']}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Update session
            session['resume_uploaded'] = True
            session['resume_filename'] = filename
            
            # Return success response
            return jsonify({
                "success": True,
                "message": "Resume uploaded successfully",
                "filename": filename
            })
        
        return jsonify({"success": False, "message": "File type not allowed. Use PDF, DOC, or DOCX."})
    except Exception as e:
        print(f"Error in upload_resume: {e}")
        return jsonify({"success": False, "message": "Upload failed. Please try again."})

@app.route('/get_history')
def get_history():
    try:
        chat_history = session.get('chat_history', [])
        # Ensure all data is JSON serializable
        chat_history = convert_numpy_types(chat_history)
        return jsonify({"chat_history": chat_history})
    except Exception as e:
        print(f"Error in get_history: {e}")
        return jsonify({"chat_history": [], "error": "Failed to get chat history"})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
