"""
Quick test script to verify fixes
"""
import os
import sys

def test_services():
    print("🔧 Testing Fixed Services...")
    print("=" * 50)
    
    # Test imports
    try:
        from services.mysql_service import MySQLService
        print("✅ MySQL service import OK")
    except Exception as e:
        print(f"❌ MySQL service import failed: {e}")
        return
    
    try:
        from services.supabase_service import SupabaseService
        print("✅ Supabase service import OK")
    except Exception as e:
        print(f"❌ Supabase service import failed: {e}")
        return
    
    try:
        from services.face_service import FaceService
        print("✅ Face service import OK")
    except Exception as e:
        print(f"❌ Face service import failed: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Test MySQL
    print("🗄️ Testing MySQL...")
    try:
        mysql_service = MySQLService()
        stats = mysql_service.get_statistics()
        if stats:
            print(f"✅ MySQL OK - Stats: {stats}")
        else:
            print("⚠️ MySQL connected but no stats returned")
    except Exception as e:
        print(f"❌ MySQL test failed: {e}")
    
    # Test Supabase
    print("\n🌐 Testing Supabase...")
    try:
        supabase_service = SupabaseService()
        if supabase_service.test_connection():
            images = supabase_service.get_recent_images(1)
            print(f"✅ Supabase OK - Found {len(images)} images")
        else:
            print("❌ Supabase connection test failed")
    except Exception as e:
        print(f"❌ Supabase test failed: {e}")
    
    # Test Face Service
    print("\n👤 Testing Face Service...")
    try:
        face_service = FaceService()
        print("✅ Face service initialized OK")
    except Exception as e:
        print(f"❌ Face service test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed! Check results above.")

if __name__ == "__main__":
    test_services()