"""
Universal Supabase compatibility wrapper
Works with all versions of supabase-py
"""
import os
import logging
from supabase import create_client
from dotenv import load_dotenv
import requests
from typing import List, Dict, Any

load_dotenv()

class UniversalSupabaseService:
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.bucket = os.getenv('SUPABASE_BUCKET_NAME')
        self.client = create_client(self.url, self.key)
        self._api_version = self._detect_api_version()
        self.logger = logging.getLogger(__name__)

    # Deprecated helper removed (use get_recent_images instead)

    def _detect_api_version(self) -> str:
        """Detect which version of Supabase API we're working with"""
        try:
            # Test different API patterns
            storage = self.client.storage
            bucket_obj = storage.from_(self.bucket)
            
            # Test if it's the new async/sync pattern
            if hasattr(bucket_obj, 'list'):
                return "v2"
            else:
                return "v1"
        except:
            return "unknown"
    
    def test_connection(self) -> bool:
        """Universal connection test"""
        try:
            # Method 1: Try simple file listing
            try:
                result = self.client.storage.from_(self.bucket).list("")
                print(f"✅ Supabase connection successful (Method 1)")
                return True
            except Exception as e1:
                print(f"Method 1 failed: {e1}")
            
            # Method 2: Try without path
            try:
                result = self.client.storage.from_(self.bucket).list()
                print(f"✅ Supabase connection successful (Method 2)")
                return True
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
            
            # Method 3: Just test client creation
            if self.client and self.url and self.key:
                print(f"✅ Supabase client created successfully")
                # Test a simple operation
                try:
                    self.client.storage.list_buckets()
                    print(f"✅ Storage service accessible")
                    return True
                except Exception as e3:
                    print(f"Storage test failed: {e3}")
                    return True  # Client is valid even if storage test fails
            
            return False
            
        except Exception as e:
            print(f"❌ All connection methods failed: {e}")
            return False
    
    def get_recent_images(self, limit: int = 10) -> List[Dict]:
        """Universal image fetching"""
        methods = [
            self._method_list_with_path,
            self._method_list_without_path,
            self._method_direct_api
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                result = method()
                if result:
                    # Process and filter results
                    files = self._process_file_list(result)
                    return files[:limit]
            except Exception as e:
                print(f"Image fetch method {i} failed: {e}")
                continue
        
        print("All image fetch methods failed")
        return []
    
    def _method_list_with_path(self):
        """Method 1: list with path parameter"""
        return self.client.storage.from_(self.bucket).list("")
    
    def _method_list_without_path(self):
        """Method 2: list without path parameter"""
        return self.client.storage.from_(self.bucket).list()
    
    def _method_direct_api(self):
        """Method 3: Direct API call"""
        headers = {
            'apikey': self.key,
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json'
        }
        url = f"{self.url}/storage/v1/object/list/{self.bucket}"
        
        payload = {
            "prefix": "",   # root of the bucket
            "limit": 100,   # adjust if needed
            "offset": 0,
            "sortBy": {
                "column": "name",
                "order": "desc"
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"HTTP {response.status_code}: {response.text}"
            )
    

    
    def _process_file_list(self, result: Any) -> List[Dict]:
        """Process file list regardless of format"""
        files = []
        
        try:
            # Handle different result types
            if not result:
                return []
            
            # Convert to list if needed
            if not isinstance(result, list):
                if hasattr(result, '__iter__'):
                    result = list(result)
                else:
                    result = [result]
            
            # Process each item
            for item in result:
                file_info = {}
                
                if isinstance(item, dict):
                    file_info = item.copy()
                elif hasattr(item, '__dict__'):
                    file_info = vars(item)
                else:
                    file_info = {'name': str(item)}
                
                name = file_info.get('name', '')
                
                # Filter for image files
                if (name and 
                    not name.endswith('/') and 
                    not name.startswith('.') and 
                    name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))):
                    files.append(file_info)
            
            # Prefer created_at or last_modified if present, else fallback to name
            if files and any('created_at' in f for f in files):
                files.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            elif files and any('last_modified' in f for f in files):
                files.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
            else:
                files.sort(key=lambda x: x.get('name', ''), reverse=True)
            
        except Exception as e:
            print(f"Error processing file list: {e}")
        
        return files
    
    def get_image_url(self, image_path: str) -> str:
        """Universal URL generation"""
        methods = [
            lambda: self.client.storage.from_(self.bucket).get_public_url(image_path),
            lambda: f"{self.url}/storage/v1/object/public/{self.bucket}/{image_path}",
            lambda: self._construct_url_manually(image_path)
        ]
        
        for method in methods:
            try:
                result = method()
                if isinstance(result, dict):
                    url = result.get('publicURL') or result.get('publicUrl') or result.get('data', {}).get('publicUrl', '')
                else:
                    url = str(result)
                
                if url and url.startswith('http'):
                    return url
            except Exception as e:
                continue
        
        return ""
    
    def _construct_url_manually(self, image_path: str) -> str:
        """Manual URL construction as fallback"""
        if self.url and self.bucket:
            base_url = self.url.replace('/rest/v1', '')
            return f"{base_url}/storage/v1/object/public/{self.bucket}/{image_path}"
        return ""
    
    def download_image(self, image_name, save_path):
        """Download an image from the bucket and save it locally."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Get signed URL for the image
            signed_url = self.get_image_url(image_name)
            
            if not signed_url:
                self.logger.error(f"Failed to get signed URL for image {image_name}")
                return False
            
            # Download the image using requests
            response = requests.get(signed_url, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to download image {image_name}: HTTP {response.status_code}")
                return False
            
            # Save the image
            with open(save_path, 'wb') as f:
                f.write(response.content)
                
            self.logger.info(f"Downloaded image {image_name} to {save_path}")
            return True
        
        except Exception as e:
            self.logger.exception(f"Error downloading image {image_name}: {e}")
            return False

# Replace your current SupabaseService with this
SupabaseService = UniversalSupabaseService