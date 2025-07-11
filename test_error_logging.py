#!/usr/bin/env python3
"""
Test script to simulate different types of errors and verify error logging
"""

import sqlite3
import os
from datetime import datetime

def test_error_logging():
    """Test error logging functionality"""
    
    # Connect to database
    db_path = 'Super-Resolution-main/users_demo.db'
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get a test user
    cursor.execute('SELECT id FROM users LIMIT 1')
    user = cursor.fetchone()
    if not user:
        print('No users found, creating a test user...')
        cursor.execute('''INSERT INTO users (name, username, password, email, is_verified, is_admin, is_approved) 
                         VALUES (?, ?, ?, ?, 1, 0, 1)''', 
                         ('Test User', 'test@example.com', 'hashed_password', 'test@example.com'))
        conn.commit()
        user_id = cursor.lastrowid
    else:
        user_id = user[0]
    
    print(f'Using user ID: {user_id}')
    
    # Test different types of errors
    error_scenarios = [
        {
            'filename': 'invalid_format.txt',
            'error_message': 'Invalid file type. Please upload an image file. Allowed types: png, jpg, jpeg, gif, bmp',
            'status': 'error'
        },
        {
            'filename': 'corrupted_image.jpg',
            'error_message': 'Error during image enhancement: Cannot identify image file',
            'status': 'error'
        },
        {
            'filename': 'no_face_detected.jpg',
            'error_message': 'No face detected in the image. Please upload an image with a clear face.',
            'status': 'error'
        },
        {
            'filename': 'file_too_large.jpg',
            'error_message': 'File size too large. Maximum allowed size: 16.0MB',
            'status': 'error'
        },
        {
            'filename': 'tensorflow_error.jpg',
            'error_message': 'Error during image enhancement: TensorFlow model failed to load',
            'status': 'error'
        },
        {
            'filename': 'memory_error.jpg',
            'error_message': 'Server error during file processing: Out of memory',
            'status': 'error'
        },
        {
            'filename': 'genai_timeout.jpg',
            'error_message': 'Error during GENAI enhancement: Request timeout after 30 seconds',
            'status': 'error'
        }
    ]
    
    # Insert test error scenarios
    for scenario in error_scenarios:
        cursor.execute('''INSERT INTO user_files 
                         (user_id, original_filename, file_size, status, error_message) 
                         VALUES (?, ?, ?, ?, ?)''', 
                        (user_id, scenario['filename'], 1024, scenario['status'], scenario['error_message']))
    
    conn.commit()
    
    # Verify errors were inserted
    cursor.execute('SELECT COUNT(*) FROM user_files WHERE error_message IS NOT NULL')
    error_count = cursor.fetchone()[0]
    print(f'Total files with error messages: {error_count}')
    
    # Display sample errors
    cursor.execute('''SELECT original_filename, status, error_message 
                     FROM user_files 
                     WHERE error_message IS NOT NULL 
                     ORDER BY upload_time DESC 
                     LIMIT 10''')
    errors = cursor.fetchall()
    
    print('\nSample error messages:')
    print('-' * 80)
    for error in errors:
        print(f'File: {error[0]}')
        print(f'Status: {error[1]}')
        print(f'Error: {error[2]}')
        print('-' * 80)
    
    conn.close()
    print(f'\nTest completed! Added {len(error_scenarios)} error scenarios.')
    print('You can now check the admin panel to see how these errors are displayed.')

if __name__ == '__main__':
    test_error_logging() 