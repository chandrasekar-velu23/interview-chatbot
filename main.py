import os
import random
import datetime
import json
import re
import secrets
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, session, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize Flask app
app = Flask(__name__)
# Generate a secure random secret key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Ensure required directories exist
os.makedirs('data', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to track session state
QUESTIONS_FILE = os.path.join(os.getcwd(), 'questions.csv')
RESULTS_FILE = os.path.join(os.getcwd(), 'data', 'interview_results.xlsx')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

# Helper function to convert numpy types to Python native types
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

# Function to generate creative responses based on user input and context
def generate_creative_response(user_message, context, step):
    """Generate a creative response based on user message, context, and current step"""
    # Default responses if NLTK fails
    thank_you_phrases = [
        "Thank you for sharing that! I appreciate your insights.",
        "That's really valuable information, thank you!",
        "I appreciate your thoughtful response.",
        "Thanks for letting me know about that!",
        "That's great to hear, thank you for sharing."
    ]
    
    follow_up_phrases = {
        'general': [
            "Let's dive a bit deeper into your background.",
            "I'd like to understand more about your experiences.",
            "Now, I'd like to explore another aspect of your profile.",
            "That's interesting! Let's move on to another important question."
        ],
        'occupation': [
            f"As a {context.get('selected_occupation', 'professional')}, I'd like to ask you something specific.",
            "Based on your professional background, I'm curious about something.",
            "Your experience is quite interesting! Here's something I'd like to know:",
            "Given your career path, I'm particularly interested in learning about:"
        ],
        'job_role': [
            f"With your interest in {context.get('selected_role', 'this field')}, I'd like to know:",
            "Your expertise in this area brings up an important question:",
            "Considering your specialization, I'm curious about:",
            "For someone with your skills, this next question is particularly relevant:"
        ],
        'job_type': [
            "Let's talk about your work preferences in more detail.",
            "I'd like to understand your ideal work arrangement better.",
            "Your work style preferences are important. Let me ask you about:",
            "Now, let's explore what work structure suits you best."
        ],
        'job_mode': [
            "Work environment matters a lot. Let's discuss that next.",
            "Your preference for how you work is important to understand.",
            "Let's explore your ideal working environment further.",
            "Now, I'd like to know about where you prefer to work."
        ],
        'resume_upload': [
            "Your resume will help me understand your full professional story.",
            "I'd love to see your resume to learn more about your journey.",
            "Your resume will provide valuable context to your answers.",
            "To complete your profile, I'll need your resume."
        ],
        'complete': [
            "Thank you for taking the time to complete this interview!",
            "You've provided some fantastic insights throughout our conversation.",
            "I've really enjoyed learning about your background and aspirations.",
            "This has been a productive conversation! I appreciate your detailed responses."
        ]
    }
    
    # Try to analyze sentiment with NLTK if available, otherwise assign neutral sentiment
    sentiment_compound = 0
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(user_message)
        sentiment_compound = sentiment['compound']
    except:
        # If NLTK fails, just use a neutral sentiment
        pass
    
    # Select appropriate response templates based on sentiment
    if sentiment_compound > 0.3:
        thank_you = "Excellent! " + random.choice(thank_you_phrases)
    elif sentiment_compound > 0:
        thank_you = random.choice(thank_you_phrases)
    elif sentiment_compound > -0.3:
        thank_you = "I understand. Thanks for sharing your perspective."
    else:
        thank_you = "I appreciate you sharing that with me, even though it might be challenging."
    
    # Generate appropriate follow-up based on current step
    if step in follow_up_phrases:
        follow_up = random.choice(follow_up_phrases[step])
    else:
        follow_up = random.choice(follow_up_phrases['general'])
    
    # Special handling for specific context keywords
    experience_words = ['experience', 'worked', 'job', 'project']
    education_words = ['learn', 'education', 'study', 'university', 'college']
    
    # Tokenize if NLTK is available
    try:
        from nltk.tokenize import word_tokenize
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        tokens = word_tokenize(user_message.lower())
        tokens = [token for token in tokens if token.isalnum()]
        
        if any(word in tokens for word in experience_words):
            follow_up = "Your experience is valuable! " + follow_up
        
        if any(word in tokens for word in education_words):
            follow_up = "Your educational background provides great context. " + follow_up
    except:
        # If NLTK fails, just check if words are in the message
        if any(word in user_message.lower() for word in experience_words):
            follow_up = "Your experience is valuable! " + follow_up
        
        if any(word in user_message.lower() for word in education_words):
            follow_up = "Your educational background provides great context. " + follow_up
    
    # Combine phrases for final response
    response = f"{thank_you} {follow_up}"
    
    return response

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
    except Exception as e:
        print(f"Error generating ID: {e}")
        return int(datetime.datetime.now().timestamp())

def get_general_questions():
    """Get the initial general questions"""
    default_questions = [
        "What is your name?",
        "Could you tell me about your educational background?",
        "What are your key skills?",
        "What are your career goals?"
    ]
    
    try:
        if os.path.exists(QUESTIONS_FILE):
            df = pd.read_csv(QUESTIONS_FILE, encoding='ISO-8859-1')
            questions = []
            for i in range(len(df)):
                question = df.iloc[i, 0]
                if pd.isna(question) or question.strip() == '':
                    break
                questions.append(question)
            return questions if questions else default_questions
        return default_questions
    except Exception as e:
        print(f"Error loading general questions: {e}")
        return default_questions

def get_occupation_questions(occupation):
    """Get occupation-specific questions"""
    default_questions = {
        'Student': [
            "What are you studying currently?",
            "When do you expect to graduate?",
            "Have you completed any internships?",
            "What courses are most relevant to your career goals?",
            "Are you involved in any extracurricular activities?"
        ],
        'Fresher': [
            "What technologies are you most familiar with?",
            "Have you worked on any projects during your education?",
            "What certifications do you have?",
            "What kind of role are you looking for?",
            "How do you stay updated with industry trends?"
        ],
        'Experienced Professional': [
            "How many years of experience do you have?",
            "What was your most challenging project?",
            "What is your current role and responsibilities?",
            "What are your leadership experiences?",
            "Why are you looking for a new opportunity?"
        ]
    }
    
    try:
        if os.path.exists(QUESTIONS_FILE):
            df = pd.read_csv(QUESTIONS_FILE, encoding='ISO-8859-1')
            occupation_columns = {
                'Student': 'Student', 
                'Fresher': 'Fresher', 
                'Experienced Professional': 'Experienced'
            }
            
            if occupation in occupation_columns:
                column = occupation_columns[occupation]
                if column in df.columns:
                    questions = df[column].dropna().tolist()
                    return random.sample(questions, min(5, len(questions))) if questions else default_questions[occupation]
        return default_questions.get(occupation, [])
    except Exception as e:
        print(f"Error loading occupation questions: {e}")
        return default_questions.get(occupation, [])

def get_job_role_questions(role):
    """Get job role specific questions"""
    default_questions = {
        "UI/UX": [
            "What design tools are you proficient with?",
            "Can you describe your design process?",
            "Have you conducted user research?",
            "What's your approach to accessibility?",
            "How do you handle stakeholder feedback?"
        ],
        "Java": [
            "What Java frameworks have you worked with?",
            "How do you handle exception management?",
            "Have you used Spring Boot?",
            "Tell me about your experience with multithreading",
            "How do you approach unit testing in Java?"
        ],
        "AI/ML": [
            "What ML libraries are you familiar with?",
            "Have you deployed models in production?",
            "What algorithms have you implemented?",
            "How do you handle model evaluation?",
            "Describe a data preprocessing challenge you've faced"
        ]
    }
    
    try:
        if os.path.exists(QUESTIONS_FILE):
            df = pd.read_csv(QUESTIONS_FILE, encoding='ISO-8859-1')
            job_roles = {"UI/UX": "UI/UX", "Java": "Java", "AI/ML": "AI/ML"}
            
            if role in job_roles:
                column = job_roles[role]
                if column in df.columns:
                    questions = df[column].dropna().tolist()
                    return random.sample(questions, min(5, len(questions))) if questions else default_questions[role]
        return default_questions.get(role, [])
    except Exception as e:
        print(f"Error loading job role questions: {e}")
        return default_questions.get(role, [])

def save_interview_data(data):
    """Save interview data to Excel file"""
    try:
        # Convert any numpy types to Python native types
        data = convert_numpy_types(data)
                
        # Create DataFrame from the data
        df = pd.DataFrame([data])
        
        # Check if file exists to append or create new
        if os.path.exists(RESULTS_FILE):
            try:
                existing_df = pd.read_excel(RESULTS_FILE)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_excel(RESULTS_FILE, index=False)
            except Exception as excel_error:
                print(f"Error reading existing Excel file: {excel_error}")
                df.to_excel(RESULTS_FILE, index=False)
        else:
            df.to_excel(RESULTS_FILE, index=False)
            
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

@app.route('/')
def index():
    # Initialize a new session
    session.clear()
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
        user_message = request.form.get('user_message', '').strip()
        
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
                # Generate creative response based on user's previous message
                if user_message:
                    bot_response = generate_creative_response(user_message, session, current_step)
                else:
                    bot_response = "Welcome to our interview chatbot! Let's get to know you better."
                
                next_question = general_questions[question_index]
                session['current_question'] = next_question
                session['question_index'] = question_index + 1
            else:
                # Move to occupation questions with a creative transition
                bot_response = generate_creative_response(user_message, session, current_step) + "\n\nNow, I'd like to understand more about your career path."
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
                        # Generate personalized response based on occupation
                        occupation_responses = {
                            'Student': f"Great to meet a fellow student! Your academic perspective is valuable.",
                            'Fresher': "Being a fresher brings a fresh perspective to the industry. I'm excited to learn more about your aspirations.",
                            'Experienced Professional': "Your professional experience is impressive! I'd love to dive deeper into your expertise."
                        }
                        bot_response = occupation_responses.get(selected_occupation, f"You selected: {selected_occupation}")
                        next_question = session['occupation_questions'][0]
                        session['current_question'] = next_question
                        session['question_index'] = 1
                    else:
                        # No occupation questions, move to job roles
                        bot_response = f"Thanks for letting me know you're a {selected_occupation}! Let's talk about your area of interest."
                        next_question = "Select a job role:"
                        is_options = True
                        options = ["UI/UX", "Java", "AI/ML"]
                        session['current_step'] = 'job_role'
                        session['current_question'] = next_question
                        session['question_index'] = 0
                else:
                    # Invalid input
                    bot_response = "I didn't quite catch that. Could you please select one of the options below?"
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
                        bot_response = generate_creative_response(user_message, session, current_step)
                        next_question = occupation_questions[question_index]
                        session['current_question'] = next_question
                        session['question_index'] = question_index + 1
                    else:
                        # Move to job roles
                        bot_response = generate_creative_response(user_message, session, current_step) + "\n\nNow, let's focus on your specific area of interest."
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
                    
                    # Personalized responses based on job role
                    role_responses = {
                        "UI/UX": "UI/UX is such a creative field! The intersection of design and user experience is fascinating.",
                        "Java": "Java development is a powerful skill set. The demand for solid Java expertise continues to grow!",
                        "AI/ML": "AI/ML is at the cutting edge of technology today. Your interest in this field shows forward thinking!"
                    }
                    
                    if session['job_role_questions']:
                        bot_response = role_responses.get(selected_role, f"You selected: {selected_role}")
                        next_question = session['job_role_questions'][0]
                        session['current_question'] = next_question
                        session['question_index'] = 1
                    else:
                        # No job role questions, move to job type
                        bot_response = role_responses.get(selected_role, f"You selected: {selected_role}") + "\n\nLet's talk about your employment preferences."
                        next_question = "Select the type of job role:"
                        is_options = True
                        options = ["Full-time", "Part-time", "Freelancing"]
                        session['current_step'] = 'job_type'
                        session['current_question'] = next_question
                        session['question_index'] = 0
                else:
                    # Invalid input
                    bot_response = "I didn't quite catch that. Could you please select one of the job roles below?"
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
                        bot_response = generate_creative_response(user_message, session, current_step)
                        next_question = job_role_questions[question_index]
                        session['current_question'] = next_question
                        session['question_index'] = question_index + 1
                    else:
                        # Move to job type
                        bot_response = generate_creative_response(user_message, session, current_step) + "\n\nNow, let's discuss what kind of employment arrangement you're looking for."
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
                
                # Personalized responses based on job type
                type_responses = {
                    "Full-time": "A full-time position offers stability and deep engagement with your team and projects.",
                    "Part-time": "Part-time work provides excellent flexibility while still maintaining professional growth.",
                    "Freelancing": "Freelancing gives you the freedom to choose diverse projects and manage your own schedule."
                }
                
                # Move to job mode
                bot_response = type_responses.get(selected_job_type, f"You selected: {selected_job_type}") + "\n\nAnd where would you prefer to work?"
                next_question = "Select your preferred job mode:"
                is_options = True
                options = ["Remote", "Onsite", "Hybrid"]
                session['current_step'] = 'job_mode'
                session['current_question'] = next_question
            else:
                # Invalid input
                bot_response = "I'm not sure I understood that choice. Please select one of the options below."
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
                
                # Personalized responses based on job mode
                mode_responses = {
                    "Remote": "Remote work offers flexibility and comfort in your own environment. Many find it boosts productivity!",
                    "Onsite": "Working onsite provides great collaboration opportunities and a clear separation between work and home.",
                    "Hybrid": "The hybrid model gives you the best of both worlds - flexibility and in-person collaboration when needed."
                }
                
                # Move to resume upload
                bot_response = mode_responses.get(selected_job_mode, f"You selected: {selected_job_mode}") + "\n\nWe're almost done! To complete your profile, I'll need your resume."
                next_question = "Please upload your resume (PDF, DOC, or DOCX format)."
                session['current_step'] = 'resume_upload'
                session['current_question'] = next_question
                is_file_upload = True
            else:
                # Invalid input
                bot_response = "I didn't quite catch that. Please select one of the work modes below."
                next_question = "Select your preferred job mode:"
                is_options = True
                options = ["Remote", "Onsite", "Hybrid"]
                session['current_question'] = next_question
        
        elif current_step == 'resume_upload':
            if session.get('resume_uploaded', False):
                # Resume already uploaded, complete the interview
                bot_response = "Amazing! Thank you for taking the time to complete this interview. Your responses and resume provide valuable insights into your background and aspirations. We'll review your information carefully!"
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
                for key, value in session.get('responses', {}).items():
                    data[f"answer_{key}"] = value
                
                for key, value in session.get('questions', {}).items():
                    data[f"question_{key}"] = value
                    
                # Save to Excel
                save_interview_data(data)
            else:
                # Waiting for resume upload
                bot_response = "I'm looking forward to seeing your resume! Please use the file upload button to share it with me (PDF, DOC, or DOCX format)."
                next_question = "Please upload your resume (PDF, DOC, or DOCX format)."
                is_file_upload = True
        
        elif current_step == 'complete':
            # Handle restart or end
            if user_message.lower() == 'yes' or user_message == '1':
                # Restart interview
                session.clear()
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
                    bot_response = "Great! I'm excited to start a new interview with you. Let's begin with getting to know you better."
                    next_question = general_questions[0]
                    session['current_question'] = next_question
                    session['question_index'] = 1
                else:
                    bot_response = "I'm sorry, but I'm having trouble accessing the questions right now. Please try again later or contact support."
            else:
                # End the chat
                bot_response = "Thank you for using our interview chatbot today! It was a pleasure getting to know you. I wish you the best of luck in your career journey! Feel free to return whenever you'd like to start another interview."
                next_question = ""
        
        # Update chat history
        chat_history = session.get('chat_history', [])
        
        if user_message:
            chat_history.append({"sender": "user", "message": user_message})
        
        if bot_response:
            chat_history.append({"sender": "bot", "message": bot_response})
        
        session['chat_history'] = chat_history
        
        # Return response
        return jsonify({
            "bot_response": bot_response,
            "next_question": next_question,
            "is_options": is_options,
            "options": options,
            "is_file_upload": is_file_upload
        })
    
    except Exception as e:
        import traceback
        print(f"Error in get_message: {e}")
        print(traceback.format_exc())
        # Return a friendly error message
        return jsonify({
            "bot_response": "I'm sorry, I encountered an issue processing your request. Let's try again.",
            "next_question": "Could you please repeat your last message?",
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
        
        # If the user does not select a file, the browser submits an empty file without a filename
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
        
        return jsonify({"success": False, "message": "File type not allowed. Please upload PDF, DOC, or DOCX file."})
    except Exception as e:
        print(f"Error in upload_resume: {e}")
        return jsonify({"success": False, "message": "An error occurred while uploading your resume. Please try again."})

@app.route('/get_history')
def get_history():
    try:
        chat_history = session.get('chat_history', [])
        # Ensure all data is JSON serializable
        chat_history = convert_numpy_types(chat_history)
        return jsonify({"chat_history": chat_history})
    except Exception as e:
        print(f"Error in get_history: {e}")
        return jsonify({"chat_history": [], "error": "Failed to retrieve chat history"})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
