#!/usr/bin/env python3
"""
Debug script to test template rendering
"""

import sqlite3
import sys
import os

# Add the Super-Resolution-main directory to the path
sys.path.insert(0, 'Super-Resolution-main')

def test_template_data():
    """Test what data is being passed to template"""
    
    # Connect to database
    db_path = 'Super-Resolution-main/users_demo.db'
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This makes results behave like dictionaries
    
    # Test the exact query from admin_panel
    per_page = 50
    file_page = 1
    
    files = conn.execute('''
        SELECT uf.id, uf.user_id, uf.original_filename, uf.enhanced_filename, 
               uf.upload_time, uf.enhancement_time, uf.file_size, uf.status, uf.error_message,
               u.email, u.username
        FROM user_files uf
        JOIN users u ON uf.user_id = u.id
        ORDER BY uf.upload_time DESC
        LIMIT ? OFFSET ?
    ''', (per_page, (file_page-1)*per_page)).fetchall()
    
    print(f"Template data test:")
    print(f"Total files: {len(files)}")
    print(f"Files with error messages: {len([f for f in files if f['error_message']])}")
    
    print("\nFirst 10 files as they would appear in template:")
    for i, file in enumerate(files[:10]):
        print(f"\nFile {i+1}:")
        print(f"  ID: {file['id']}")
        print(f"  Filename: {file['original_filename']}")
        print(f"  Status: {file['status']}")
        print(f"  Error message: {file['error_message']}")
        print(f"  Template condition (file.error_message): {bool(file['error_message'])}")
        
        # Test the template condition logic
        if file['error_message']:
            print(f"  ✅ ERROR DETAILS SHOULD BE SHOWN")
            print(f"  Error text: {file['error_message'][:100]}...")
        else:
            print(f"  ❌ No error details to show")
    
    conn.close()

if __name__ == '__main__':
    test_template_data() 