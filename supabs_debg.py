import os
from dotenv import load_dotenv

load_dotenv()

def debug_supabase_config():
    print("🔍 Supabase Configuration Debug")
    print("=" * 50)
    
    # Check environment variables
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY') 
    bucket = os.getenv('SUPABASE_BUCKET')
    
    print(f"URL: {url[:50] + '...' if url and len(url) > 50 else url or 'NOT SET'}")
    print(f"Key: {'SET (' + str(len(key)) + ' chars)' if key else 'NOT SET'}")
    print(f"Bucket: {bucket or 'NOT SET'}")
    
    # Check if all required vars are set
    missing = []
    if not url: missing.append('SUPABASE_URL')
    if not key: missing.append('SUPABASE_KEY') 
    if not bucket: missing.append('SUPABASE_BUCKET')
    
    if missing:
        print(f"\n❌ Missing variables: {', '.join(missing)}")
        return False
    
    print("\n✅ All environment variables are set")
    
    # Test URL format
    if not url.startswith('https://'):
        print("⚠️ URL should start with https://")
    
    if not url.endswith('.supabase.co'):
        print("⚠️ URL should end with .supabase.co")
    
    # Test key format (should be long and start with specific prefix)
    if len(key) < 100:
        print("⚠️ API key seems too short")
    
    return True

def test_supabase_manually():
    """Manual test without client library"""
    import requests
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    bucket = os.getenv('SUPABASE_BUCKET')
    
    print("\n🧪 Manual API Test")
    print("=" * 30)
    
    # Test 1: List buckets
    try:
        headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}'
        }
        
        buckets_url = f"{url}/storage/v1/bucket"
        response = requests.get(buckets_url, headers=headers, timeout=10)
        
        print(f"Bucket list: HTTP {response.status_code}")
        if response.status_code == 200:
            buckets = response.json()
            bucket_names = [b['name'] for b in buckets]
            print(f"Available buckets: {bucket_names}")
            
            if bucket in bucket_names:
                print(f"✅ Bucket '{bucket}' exists")
            else:
                print(f"❌ Bucket '{bucket}' not found")
                return False
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Bucket test failed: {e}")
        return False
    
    # Test 2: List files in bucket
    try:
        files_url = f"{url}/storage/v1/object/list/{bucket}"
        response = requests.post(files_url, headers=headers, json={}, timeout=10)
        
        print(f"File list: HTTP {response.status_code}")
        if response.status_code == 200:
            files = response.json()
            print(f"Found {len(files)} items in bucket")
            
            image_files = [f for f in files if f.get('name', '').lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            print(f"Image files: {len(image_files)}")
            
            if image_files:
                print("Sample files:")
                for f in image_files[:3]:
                    print(f"  - {f.get('name')}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ File list test failed: {e}")
        return False

if __name__ == "__main__":
    if debug_supabase_config():
        test_supabase_manually()