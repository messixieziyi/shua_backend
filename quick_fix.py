import re

# Read the current file
with open('split_interface_final_fixed.html', 'r') as f:
    content = f.read()

# Check if the generateRandomEvent function is still there
if 'function generateRandomEvent()' not in content:
    print("❌ generateRandomEvent function is missing!")
else:
    print("✅ generateRandomEvent function exists")

# Check if the requestToJoin function is still there
if 'function requestToJoin()' not in content:
    print("❌ requestToJoin function is missing!")
else:
    print("✅ requestToJoin function exists")

# Check if the window.onload is still there
if 'window.onload' not in content:
    print("❌ window.onload is missing!")
else:
    print("✅ window.onload exists")

print("File length:", len(content))
