#!/usr/bin/env python3
"""Test Cloudinary configuration"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Test Cloudinary configuration
from cloudinary_config import is_cloudinary_configured, upload_image

print("=" * 60)
print("CLOUDINARY CONFIGURATION TEST")
print("=" * 60)

# Check if configured
if is_cloudinary_configured():
    print("✅ Cloudinary is CONFIGURED!")
    print(f"\nCloud Name: {os.getenv('CLOUDINARY_CLOUD_NAME')}")
    print(f"API Key: {os.getenv('CLOUDINARY_API_KEY')[:10]}..." if os.getenv('CLOUDINARY_API_KEY') else "API Key: NOT SET")
    print(f"API Secret: {'*' * 20}..." if os.getenv('CLOUDINARY_API_SECRET') else "API Secret: NOT SET")
    
    # Test upload with a tiny 1x1 pixel image
    print("\n" + "=" * 60)
    print("TESTING IMAGE UPLOAD")
    print("=" * 60)
    
    # Tiny 1x1 red pixel PNG (base64)
    test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    print("\nUploading test image (1x1 pixel)...")
    success, result = upload_image(test_image, folder="shua/test")
    
    if success:
        print(f"✅ Upload successful!")
        print(f"\nCloudinary URL: {result}")
        print("\n🎉 Everything is working! You can now upload images through your app.")
    else:
        print(f"❌ Upload failed: {result}")
        
else:
    print("❌ Cloudinary is NOT configured")
    print("\nMissing environment variables:")
    if not os.getenv('CLOUDINARY_CLOUD_NAME'):
        print("  - CLOUDINARY_CLOUD_NAME")
    if not os.getenv('CLOUDINARY_API_KEY'):
        print("  - CLOUDINARY_API_KEY")
    if not os.getenv('CLOUDINARY_API_SECRET'):
        print("  - CLOUDINARY_API_SECRET")
    print("\nPlease add these to your .env file and try again.")

print("=" * 60)
