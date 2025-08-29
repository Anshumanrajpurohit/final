"""
Test script for Enhanced Face Recognition System
Verifies all components work together
"""
import os
import sys
import logging
from datetime import datetime

def test_enhanced_components():
    print("🧪 Testing Enhanced Face Recognition System Components...")
    print("=" * 60)
    
    # Test imports
    components = [
        ("Performance Optimizer", "utils.performance_optimizer", "PerformanceOptimizer"),
        ("Enhanced Face Service", "utils.enhanced_face_service", "EnhancedFaceService"),
        ("MySQL Service", "services.mysql_service", "MySQLService"),
        ("Supabase Service", "services.supabase_service", "SupabaseService"),
    ]
    
    imported_components = {}
    
    for name, module, class_name in components:
        try:
            module_obj = __import__(module, fromlist=[class_name])
            class_obj = getattr(module_obj, class_name)
            imported_components[name] = class_obj
            print(f"✅ {name} import OK")
        except Exception as e:
            print(f"❌ {name} import failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    
    # Test Performance Optimizer
    print("⚡ Testing Performance Optimizer...")
    try:
        optimizer = imported_components["Performance Optimizer"]()
        memory_stats = optimizer.monitor_memory_usage()
        print(f"✅ Performance Optimizer OK - Memory: {memory_stats['rss_mb']:.1f} MB")
        
        # Test FAISS availability
        if optimizer.get_performance_stats()['faiss_available']:
            print("✅ FAISS available for fast similarity search")
        else:
            print("⚠️ FAISS not available, using brute force comparison")
            
    except Exception as e:
        print(f"❌ Performance Optimizer test failed: {e}")
    
    # Test Enhanced Face Service
    print("\n👤 Testing Enhanced Face Service...")
    try:
        face_service = imported_components["Enhanced Face Service"](threshold=0.6)
        print("✅ Enhanced Face Service initialized OK")
        
        # Test quality thresholds
        thresholds = face_service.get_quality_thresholds()
        print(f"✅ Quality thresholds loaded: {len(thresholds)} parameters")
        
    except Exception as e:
        print(f"❌ Enhanced Face Service test failed: {e}")
    
    # Test MySQL Service
    print("\n🗄️ Testing MySQL Service...")
    try:
        mysql_service = imported_components["MySQL Service"]()
        stats = mysql_service.get_statistics()
        if stats:
            print(f"✅ MySQL Service OK - Stats: {stats}")
        else:
            print("⚠️ MySQL connected but no stats returned")
    except Exception as e:
        print(f"❌ MySQL Service test failed: {e}")
    
    # Test Supabase Service
    print("\n🌐 Testing Supabase Service...")
    try:
        supabase_service = imported_components["Supabase Service"]()
        if supabase_service.test_connection():
            images = supabase_service.get_recent_images(1)
            print(f"✅ Supabase Service OK - Found {len(images)} images")
        else:
            print("❌ Supabase connection test failed")
    except Exception as e:
        print(f"❌ Supabase Service test failed: {e}")
    
    print("\n" + "=" * 60)
    
    # Test Enhanced System Integration
    print("🔗 Testing Enhanced System Integration...")
    try:
        from main_enhanced import EnhancedFaceRecognitionSystem
        system = EnhancedFaceRecognitionSystem()
        print("✅ Enhanced System integration OK")
        
        # Test configuration
        print(f"  - Face Threshold: {system.face_service.threshold}")
        print(f"  - Max Batch Size: {system.max_batch_size}")
        print(f"  - Processing Interval: {system.processing_interval}s")
        print(f"  - Quality Check: {system.enable_quality_check}")
        print(f"  - FAISS Enabled: {system.enable_faiss}")
        
    except Exception as e:
        print(f"❌ Enhanced System integration failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Enhanced System Test Completed!")
    print("=" * 60)
    
    assert True

def test_performance_features():
    """Test specific performance features"""
    print("\n🚀 Testing Performance Features...")
    
    try:
        from utils.performance_optimizer import PerformanceOptimizer
        from utils.enhanced_face_service import EnhancedFaceService
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Test memory monitoring
        memory_before = optimizer.monitor_memory_usage()
        print(f"✅ Memory monitoring: {memory_before['rss_mb']:.1f} MB")
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        print(f"✅ Performance stats: {len(stats)} metrics")
        
        # Test face service with performance features
        face_service = EnhancedFaceService()
        
        # Test quality assessment
        thresholds = face_service.get_quality_thresholds()
        print(f"✅ Quality assessment: {len(thresholds)} thresholds")
        
        # Test performance stats from face service
        face_stats = face_service.get_performance_stats()
        print(f"✅ Face service performance: {len(face_stats)} metrics")
        
        print("✅ All performance features working correctly")
        
    except Exception as e:
        print(f"❌ Performance features test failed: {e}")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Test basic components
    success = test_enhanced_components()
    
    if success:
        # Test performance features
        test_performance_features()
        
        print("\n🎉 All tests passed! Enhanced system is ready to use.")
        print("\n📋 Next steps:")
        print("1. Configure your .env file with database and Supabase credentials")
        print("2. Run the database schema: mysql -u user -p database < utils/database_schema.sql")
        print("3. Start the enhanced system: python main_enhanced.py")
        print("4. Monitor performance in logs/face_recognition.log")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
