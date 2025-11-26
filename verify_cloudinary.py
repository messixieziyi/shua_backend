#!/usr/bin/env python3
"""
Quick test to verify Cloudinary is being used when uploading images
"""

import requests
import json

# Test endpoint
BASE_URL = "http://localhost:8002"

# Small test image (1x1 red pixel)
TEST_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

print("=" * 60)
print("CLOUDINARY INTEGRATION TEST")
print("=" * 60)

# First, you need to be logged in. This test assumes you have a token.
# You can get your token from the browser's localStorage or by logging in.

print("\n⚠️  To run this test, you need:")
print("1. Backend running on http://localhost:8002")
print("2. A valid authentication token")
print("\nTo get your token:")
print("1. Open http://localhost:3000 in your browser")
print("2. Log in to your account")
print("3. Open browser DevTools (F12)")
print("4. Go to Console tab")
print("5. Type: localStorage.getItem('token')")
print("6. Copy the token value")

token = input("\nPaste your auth token here (or press Enter to skip): ").strip()

if not token:
    print("\n⚠️  Skipping API test. Testing Cloudinary config only...")
    
    # Just test if cloudinary_config is importable
    try:
        import sys
        sys.path.insert(0, '/Users/estheryu/Desktop/shua_backend')
        from cloudinary_config import is_cloudinary_configured
        
        if is_cloudinary_configured():
            print("✅ Cloudinary is configured!")
            print("\nTo verify it's being used:")
            print("1. Restart your backend server")
            print("2. Upload an image through the frontend")
            print("3. Check the database - the image URL should start with:")
            print("   https://res.cloudinary.com/")
        else:
            print("❌ Cloudinary is NOT configured")
            print("Check your .env file for CLOUDINARY_* variables")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("\n🔍 Testing image upload...")
    
    # Test profile picture upload
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "profile_picture": TEST_IMAGE
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/users/profile-picture",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            profile_picture_url = result.get('profile_picture', '')
            
            print(f"\n✅ Upload successful!")
            print(f"\nReturned URL: {profile_picture_url[:100]}...")
            
            if profile_picture_url.startswith('https://res.cloudinary.com/'):
                print("\n🎉 SUCCESS! Cloudinary is being used!")
                print(f"Your image is stored at: {profile_picture_url}")
            elif profile_picture_url.startswith('data:image/'):
                print("\n❌ FAIL! Still using base64 (not Cloudinary)")
                print("This means the backend server needs to be restarted.")
            else:
                print(f"\n⚠️  Unexpected URL format: {profile_picture_url}")
        else:
            print(f"\n❌ Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Backend is running on http://localhost:8002")
        print("2. Your token is valid")

print("\n" + "=" * 60)
