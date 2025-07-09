#!/usr/bin/env python3
"""
Demo version of the Super Resolution Web Application
This version uses mock responses to test the frontend without requiring AI models
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from flask_cors import CORS
import os
import uuid
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import psutil
import time
import threading
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import shutil

# Import system monitoring
from system_monitor import system_monitor

# --- Import real enhancement logic ---
from super_resol import enhance_image_api_method
import tensorflow as tf
# --- Import PDF utility ---
from utility import create_pdf
from genai_enhance import genai_enhance_image_api_method

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['DATABASE'] = 'users_demo.db'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    # Add new fields: name, department, branch_location, mobile
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT DEFAULT '',
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        is_verified INTEGER DEFAULT 0,
        is_admin INTEGER DEFAULT 0,
        is_approved INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        password_reset_requested INTEGER DEFAULT 0,
        password_reset_date TIMESTAMP,
        department TEXT DEFAULT '',
        branch_location TEXT DEFAULT '',
        mobile TEXT DEFAULT ''
    )''')
    # Add user_activity table
    conn.execute('''CREATE TABLE IF NOT EXISTS user_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        event_type TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        session_id TEXT,
        ip_address TEXT
    )''')
    # Add user_files table
    conn.execute('''CREATE TABLE IF NOT EXISTS user_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        original_filename TEXT,
        enhanced_filename TEXT,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        enhancement_time TIMESTAMP,
        file_size INTEGER,
        status TEXT DEFAULT 'uploaded'
    )''')
    # Try to add columns if upgrading from old schema
    try:
        conn.execute('ALTER TABLE users ADD COLUMN name TEXT DEFAULT ""')
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE users ADD COLUMN department TEXT DEFAULT ""')
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE users ADD COLUMN branch_location TEXT DEFAULT ""')
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE users ADD COLUMN mobile TEXT DEFAULT ""')
    except Exception:
        pass
    # Update existing users to be approved if they don't have the is_approved field set
    try:
        conn.execute('UPDATE users SET is_approved = 1 WHERE is_approved IS NULL')
    except Exception:
        pass
    # Create default admin user if it doesn't exist
    admin_email = 'admin@superresolution.com'
    admin_password = 'admin123'  # You should change this in production
    existing_admin = conn.execute('SELECT * FROM users WHERE email = ?', (admin_email,)).fetchone()
    if not existing_admin:
        conn.execute('''INSERT INTO users (name, username, password, email, is_verified, is_admin, is_approved) 
                       VALUES (?, ?, ?, ?, 1, 1, 1)''',
                    ("Admin", admin_email, generate_password_hash(admin_password), admin_email))
        print(f"✅ Created default admin user: {admin_email} / {admin_password}")
    conn.commit()
    conn.close()

init_db()  # Ensure tables are created on every start

def log_user_activity(user_id, event_type, session_id=None, ip_address=None):
    print(f"DEBUG: Logging activity - User: {user_id}, Event: {event_type}, Session: {session_id}, IP: {ip_address}")
    conn = get_db()
    conn.execute(
        'INSERT INTO user_activity (user_id, event_type, session_id, ip_address) VALUES (?, ?, ?, ?)',
        (user_id, event_type, session_id, ip_address)
    )
    conn.commit()
    conn.close()
    print(f"DEBUG: Activity logged successfully")

def log_file_activity(user_id, original_filename, file_size, status='uploaded'):
    """Log file upload"""
    print(f"DEBUG: Logging file upload - User: {user_id}, File: {original_filename}, Size: {file_size}")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO user_files (user_id, original_filename, file_size, status) VALUES (?, ?, ?, ?)',
        (user_id, original_filename, file_size, status)
    )
    conn.commit()
    file_id = cursor.lastrowid
    conn.close()
    print(f"DEBUG: File upload logged with ID: {file_id}")
    return file_id

def update_file_enhancement(file_id, enhanced_filename, status='enhanced'):
    """Update file record when enhancement is complete"""
    print(f"DEBUG: Updating file enhancement - File ID: {file_id}, Enhanced: {enhanced_filename}")
    conn = get_db()
    conn.execute(
        'UPDATE user_files SET enhanced_filename = ?, enhancement_time = CURRENT_TIMESTAMP, status = ? WHERE id = ?',
        (enhanced_filename, status, file_id)
    )
    conn.commit()
    conn.close()
    print(f"DEBUG: File enhancement updated")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html', username=session.get('username'))
    else:
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        department = request.form['department']
        branch_location = request.form['branch_location']
        mobile = request.form['mobile']
        print("Signup values:", name, department, branch_location, mobile)  # DEBUG
        if not name or not password or not email or not department or not branch_location or not mobile:
            return render_template('signup.html', error='Please fill in all fields.')
        conn = get_db()
        try:
            conn.execute('''INSERT INTO users 
                (name, username, password, email, is_verified, is_admin, is_approved, department, branch_location, mobile)
                VALUES (?, ?, ?, ?, 1, 0, 0, ?, ?, ?)''',
                (name, email, generate_password_hash(password), email, department, branch_location, mobile))
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.close()
            if 'email' in str(e):
                return render_template('signup.html', error='Email already exists.')
            return render_template('signup.html', error='Account already exists.')
        conn.close()
        return render_template('signup.html', message='Account created! Please wait for admin approval before logging in.')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        print(f"DEBUG: Login attempt for email: {email}")
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            if not user['is_approved'] and not user['is_admin']:
                return render_template('login.html', error='Your account is pending admin approval. Please wait.')
            session['user_id'] = user['id']
            session['username'] = user['email']
            session['is_admin'] = user['is_admin']
            print(f"DEBUG: Login successful for user ID: {user['id']}")
            # Log login event
            try:
                log_user_activity(user['id'], 'login', session_id=str(session.get('_id')), ip_address=request.remote_addr)
                print(f"DEBUG: Login activity logged")
            except Exception as e:
                print(f"DEBUG: Error logging login activity: {e}")
            return redirect(url_for('index'))
        else:
            print(f"DEBUG: Login failed for email: {email}")
            return render_template('login.html', error='Invalid email or password.')
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user:
            # Mark that password reset was requested
            conn.execute('''UPDATE users 
                           SET password_reset_requested = 1, password_reset_date = CURRENT_TIMESTAMP 
                           WHERE email = ?''', (email,))
            conn.commit()
            conn.close()
            return render_template('forgot_password.html', 
                                 message='Password reset request submitted. An admin will review and set a new password for you.')
        else:
            conn.close()
            return render_template('forgot_password.html', 
                                 error='Email not found. Please check your email address.')
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    print(f"DEBUG: Logout attempt for user_id: {session.get('user_id')}")
    # Log logout event before clearing session
    if 'user_id' in session:
        try:
            log_user_activity(session['user_id'], 'logout', session_id=str(session.get('_id')), ip_address=request.remote_addr)
            print(f"DEBUG: Logout activity logged")
        except Exception as e:
            print(f"DEBUG: Error logging logout activity: {e}")
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin/data')
def admin_data():
    """Return JSON data for admin panel auto-refresh"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = get_db()
    
    # Get all users including the current admin
    users = conn.execute('''
        SELECT id, username, email, is_verified, is_admin, is_approved, created_at, 
               password_reset_requested, password_reset_date, department, branch_location, mobile
        FROM users 
        ORDER BY created_at DESC
    ''').fetchall()
    
    # Get recent user activity
    activities = conn.execute('''
        SELECT ua.id, ua.user_id, ua.event_type, ua.timestamp, ua.session_id, ua.ip_address,
               u.email, u.username
        FROM user_activity ua
        JOIN users u ON ua.user_id = u.id
        ORDER BY ua.timestamp DESC
        LIMIT 50
    ''').fetchall()
    
    # Get recent file activities
    files = conn.execute('''
        SELECT uf.id, uf.user_id, uf.original_filename, uf.enhanced_filename, 
               uf.upload_time, uf.enhancement_time, uf.file_size, uf.status,
               u.email, u.username
        FROM user_files uf
        JOIN users u ON uf.user_id = u.id
        ORDER BY uf.upload_time DESC
        LIMIT 50
    ''').fetchall()
    
    conn.close()
    
    # Convert to dictionaries for JSON serialization
    users_data = [dict(user) for user in users]
    activities_data = [dict(activity) for activity in activities]
    files_data = [dict(file) for file in files]
    
    return jsonify({
        'users': users_data,
        'activities': activities_data,
        'files': files_data
    })

@app.route('/admin')
def admin_panel():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    conn = get_db()
    # Get all users including the current admin
    users = conn.execute('''
        SELECT id, name, username, email, is_verified, is_admin, is_approved, created_at, 
               password_reset_requested, password_reset_date, department, branch_location, mobile
        FROM users 
        ORDER BY created_at DESC
    ''').fetchall()
    
    # Get recent user activity
    activities = conn.execute('''
        SELECT ua.id, ua.user_id, ua.event_type, ua.timestamp, ua.session_id, ua.ip_address,
               u.email, u.username
        FROM user_activity ua
        JOIN users u ON ua.user_id = u.id
        ORDER BY ua.timestamp DESC
        LIMIT 50
    ''').fetchall()
    
    # Get recent file activities
    files = conn.execute('''
        SELECT uf.id, uf.user_id, uf.original_filename, uf.enhanced_filename, 
               uf.upload_time, uf.enhancement_time, uf.file_size, uf.status,
               u.email, u.username
        FROM user_files uf
        JOIN users u ON uf.user_id = u.id
        ORDER BY uf.upload_time DESC
        LIMIT 50
    ''').fetchall()
    
    print(f"DEBUG: Found {len(activities)} activities")
    for activity in activities:
        print(f"DEBUG: Activity - {activity['timestamp']} - {activity['email']} - {activity['event_type']}")
    
    print(f"DEBUG: Found {len(files)} file activities")
    for file in files:
        print(f"DEBUG: File - {file['upload_time']} - {file['email']} - {file['original_filename']} - {file['status']}")
    
    conn.close()
    
    return render_template('admin.html', users=users, activities=activities, files=files)

@app.route('/admin/approve/<int:user_id>', methods=['POST'])
def approve_user(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = get_db()
    conn.execute('UPDATE users SET is_approved = 1 WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User approved successfully'})

@app.route('/admin/reject/<int:user_id>', methods=['POST'])
def reject_user(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = get_db()
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User rejected and deleted'})

@app.route('/admin/toggle-admin/<int:user_id>', methods=['POST'])
def toggle_admin(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = get_db()
    # Get current admin status
    user = conn.execute('SELECT is_admin FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        new_status = 0 if user['is_admin'] else 1
        conn.execute('UPDATE users SET is_admin = ? WHERE id = ?', (new_status, user_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'Admin status {"removed" if new_status == 0 else "granted"}'})
    
    conn.close()
    return jsonify({'error': 'User not found'}), 404

@app.route('/admin/password-reset/<int:user_id>', methods=['POST'])
def admin_password_reset(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    new_password = data.get('new_password')
    
    if not new_password or len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400
    
    conn = get_db()
    user = conn.execute('SELECT email FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        conn.execute('''UPDATE users 
                       SET password = ?, password_reset_requested = 0, password_reset_date = NULL 
                       WHERE id = ?''', 
                    (generate_password_hash(new_password), user_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'Password updated for {user["email"]}'})
    
    conn.close()
    return jsonify({'error': 'User not found'}), 404

@app.route('/admin/clear-password-reset/<int:user_id>', methods=['POST'])
def clear_password_reset(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = get_db()
    user = conn.execute('SELECT email FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        conn.execute('''UPDATE users 
                       SET password_reset_requested = 0, password_reset_date = NULL 
                       WHERE id = ?''', (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'Password reset request cleared for {user["email"]}'})
    
    conn.close()
    return jsonify({'error': 'User not found'}), 404

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db()
    user = conn.execute('SELECT email FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'User {user["email"]} deleted successfully'})
    conn.close()
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized. Please log in.'}), 401
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        # Get file size
        file_size = os.path.getsize(temp_path)
        
        # Log file upload
        file_id = log_file_activity(session['user_id'], filename, file_size, 'uploaded')
        
        try:
            # --- Use the real enhancement function ---
            results = enhance_image_api_method(temp_path)
            if results is None:
                # No face detected or error
                update_file_enhancement(file_id, None, 'error')
                os.remove(temp_path)
                return jsonify({'error': 'No face detected or enhancement failed.'}), 400

            # results is a tuple of 5 PIL Images (one for each fidelity)
            enhanced_images = [image_to_base64(img) for img in results]

            # Save the original image path (already saved as temp_path)
            saved_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{filename}")
            shutil.copy(temp_path, saved_original_path)

            # Save enhanced images to disk and collect their paths
            enhanced_image_paths = []
            for idx, img in enumerate(results):
                enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{idx}_{filename}")
                img.save(enhanced_path)
                enhanced_image_paths.append(enhanced_path)
            
            # Update file record with enhancement completion
            enhanced_filename = f"enhanced_{filename}"
            update_file_enhancement(file_id, enhanced_filename, 'enhanced')
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'enhanced_images': enhanced_images,
                'message': 'Image enhanced successfully!',
                'original_image_path': saved_original_path,
                'enhanced_image_paths': enhanced_image_paths
            })
            
        except Exception as e:
            # Update file record with error status
            update_file_enhancement(file_id, None, 'error')
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/genai-enhance', methods=['POST'])
def genai_enhance_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized. Please log in.'}), 401
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        file_id = log_file_activity(session['user_id'], filename, file_size, 'uploaded')
        try:
            result_data = genai_enhance_image_api_method(temp_path)
            if result_data is None:
                update_file_enhancement(file_id, None, 'error')
                os.remove(temp_path)
                return jsonify({'error': 'GENAI enhancement failed.'}), 400
            
            # Extract images and metadata
            enhanced_images = [image_to_base64(img) for img in result_data['images']]
            similarity_scores = result_data['similarity_scores'] 
            recommended_idx = result_data['recommended_idx']
            
            saved_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{filename}")
            shutil.copy(temp_path, saved_original_path)
            enhanced_image_paths = []
            for idx, img in enumerate(result_data['images']):
                enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"genai_enhanced_{idx}_{filename}")
                img.save(enhanced_path)
                enhanced_image_paths.append(enhanced_path)
            enhanced_filename = f"genai_enhanced_{filename}"
            update_file_enhancement(file_id, enhanced_filename, 'enhanced')
            os.remove(temp_path)
            return jsonify({
                'success': True,
                'enhanced_images': enhanced_images,
                'similarity_scores': similarity_scores,
                'recommended_idx': recommended_idx,
                'message': 'GENAI enhancement successful!',
                'original_image_path': saved_original_path,
                'enhanced_image_paths': enhanced_image_paths
            })
        except Exception as e:
            update_file_enhancement(file_id, None, 'error')
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'message': 'Super Resolution API is running (Demo Mode)',
        'mode': 'demo'
    })

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate PDF report for image enhancement"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized. Please log in.'}), 401
    
    try:
        data = request.get_json()
        original_image_path = data.get('original_image_path')
        processed_images_paths = data.get('processed_images_paths')
        filename = data.get('filename', 'enhanced_image')
        request_info = data.get('request_info', {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'taskId': 'N/A',
            'userEmail': session.get('username', 'N/A')
        })
        
        if not original_image_path or not processed_images_paths:
            return jsonify({'error': 'Missing image data'}), 400
        
        # --- File existence check ---
        import os
        missing = []
        if not os.path.exists(original_image_path):
            missing.append(original_image_path)
        for p in processed_images_paths:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            return jsonify({'error': f'Missing image files: {missing}'}), 400
        # --- End file existence check ---
        
        # Use the utility function to create the PDF
        pdf_data = create_pdf(original_image_path, processed_images_paths, request_info)
        buffer = BytesIO(pdf_data)
        
        # Return PDF as response
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"enhancement_report_{filename}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"DEBUG: Error in generate_report: {str(e)}")
        print(f"DEBUG: Original image path: {data.get('original_image_path')}")
        print(f"DEBUG: Processed images paths: {data.get('processed_images_paths')}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

@app.route('/api/monitor/start', methods=['POST'])
def start_monitoring():
    """Start system monitoring"""
    try:
        interval = request.json.get('interval', 5) if request.is_json else 5
        success = system_monitor.start_monitoring(interval)
        return jsonify({
            'success': success,
            'message': 'Monitoring started' if success else 'Monitoring already running'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/stop', methods=['POST'])
def stop_monitoring():
    """Stop system monitoring"""
    try:
        system_monitor.stop_monitoring()
        return jsonify({
            'success': True,
            'message': 'Monitoring stopped'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/current')
def get_current_stats():
    """Get current system statistics"""
    try:
        stats = system_monitor.get_current_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/history')
def get_monitoring_history():
    """Get monitoring history"""
    try:
        limit = request.args.get('limit', type=int)
        history = system_monitor.get_monitoring_history(limit)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/clear', methods=['POST'])
def clear_monitoring_history():
    """Clear monitoring history"""
    try:
        system_monitor.clear_history()
        return jsonify({
            'success': True,
            'message': 'Monitoring history cleared'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/status')
def get_monitoring_status():
    """Get monitoring status"""
    try:
        return jsonify({
            'is_monitoring': system_monitor.is_monitoring,
            'data_points': len(system_monitor.monitoring_data),
            'max_data_points': system_monitor.max_data_points
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎭 Starting Super Resolution Web Application in DEMO MODE...")
    print("📝 This is a demo version that creates mock enhanced images with user login/signup and email verification")
    print("🌐 The application will be available at: http://localhost:5001")
    print("⚠️  Note: This is for frontend testing only - no real AI processing")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False) 