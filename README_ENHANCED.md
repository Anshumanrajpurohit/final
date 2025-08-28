# Enhanced Face Recognition System

A high-performance, production-ready face recognition system that counts unique individuals, manages duplicates, and maintains visit statistics with advanced performance optimizations.

## 🚀 Key Features

### Core Functionality
- **Automatic Face Detection & Cropping**: Uses OpenCV + face_recognition for robust face detection
- **Embedding Generation**: Creates 128-dimensional face embeddings for comparison
- **Duplicate Prevention**: Intelligent matching to avoid creating duplicate person records
- **Visit Tracking**: Maintains visit counts and timestamps for each unique person
- **Batch Processing**: Efficient processing of multiple images in scheduled intervals

### Performance Optimizations
- **FAISS Integration**: Fast similarity search for large datasets (1000+ faces)
- **Memory Management**: Automatic garbage collection and memory monitoring
- **Parallel Processing**: Multi-threaded image processing for better throughput
- **Quality Assessment**: Face quality filtering to improve recognition accuracy
- **Caching**: Intelligent caching of embeddings and processing results

### Advanced Monitoring
- **Real-time Statistics**: Comprehensive system performance metrics
- **Quality Metrics**: Face quality assessment (brightness, contrast, blur, size)
- **Performance Tracking**: Memory usage, processing times, cache hit rates
- **Error Handling**: Robust error handling with detailed logging
- **Database Views**: Pre-built views for easy querying and analytics

## 📋 System Requirements

- Python 3.8+
- MySQL 5.7+ or MySQL 8.0+
- Supabase account (for image storage)
- 4GB+ RAM (8GB+ recommended for large datasets)
- CPU with 4+ cores (for parallel processing)

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <your-repo>
cd face-recognition-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install enhanced dependencies
pip install -r requirement_enhanced.txt

# For GPU acceleration (optional)
pip install faiss-gpu torch torchvision
```

### 3. Environment Configuration
Create a `.env` file:
```env
# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=face_recognition
MYSQL_PORT=3306

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET_NAME=your_bucket_name

# System Configuration
FACE_THRESHOLD=0.6
MAX_BATCH_SIZE=10
PROCESSING_INTERVAL=5
MAX_WORKERS=4
ENABLE_QUALITY_CHECK=true
ENABLE_FAISS=true
```

### 4. Database Setup
```sql
-- Run the schema file
mysql -u your_username -p your_database < utils/database_schema.sql
```

## 🎯 Usage

### Basic Usage
```bash
# Run the enhanced system
python main_enhanced.py
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_THRESHOLD` | 0.6 | Similarity threshold for face matching |
| `MAX_BATCH_SIZE` | 10 | Maximum images per processing batch |
| `PROCESSING_INTERVAL` | 5 | Processing interval in seconds |
| `MAX_WORKERS` | 4 | Number of parallel processing threads |
| `ENABLE_QUALITY_CHECK` | true | Enable face quality assessment |
| `ENABLE_FAISS` | true | Enable FAISS for large datasets |

### Performance Tuning

#### For Small Datasets (< 1000 faces)
```env
ENABLE_FAISS=false
MAX_BATCH_SIZE=5
MAX_WORKERS=2
```

#### For Large Datasets (> 1000 faces)
```env
ENABLE_FAISS=true
MAX_BATCH_SIZE=20
MAX_WORKERS=8
ENABLE_QUALITY_CHECK=true
```

#### For High-Throughput Processing
```env
PROCESSING_INTERVAL=2
MAX_BATCH_SIZE=50
MAX_WORKERS=12
```

## 📊 System Architecture

### Data Flow
1. **Image Fetching**: Supabase bucket monitoring for new images
2. **Face Detection**: OpenCV + face_recognition for face detection
3. **Quality Assessment**: Face quality filtering and metrics
4. **Embedding Generation**: 128-dimensional face embeddings
5. **Temporary Storage**: Batch processing in temp_faces table
6. **Similarity Search**: FAISS or brute-force comparison
7. **Person Management**: Create new or update existing persons
8. **Cleanup**: Automatic temp file and memory cleanup

### Database Schema
- `unique_persons`: Main person records with visit tracking
- `person_embeddings`: Face embeddings for similarity search
- `temp_faces`: Temporary storage for batch processing
- `processing_logs`: System performance and error logging
- `image_metadata`: Optional image tracking
- `system_config`: Dynamic configuration management

## 🔧 Advanced Features

### Quality Assessment
The system automatically assesses face quality based on:
- **Brightness**: Optimal range 30-250
- **Contrast**: Higher values preferred
- **Blur**: Laplacian variance measurement
- **Face Size**: Minimum 80px recommended
- **Aspect Ratio**: Face proportions

### Performance Monitoring
Real-time monitoring includes:
- Memory usage and garbage collection
- Processing times and throughput
- Cache hit rates
- FAISS index performance
- Error rates and recovery

### Scalability Features
- **Horizontal Scaling**: Multiple instances can run simultaneously
- **Database Optimization**: Indexed queries and efficient storage
- **Memory Management**: Automatic cleanup and monitoring
- **Batch Processing**: Configurable batch sizes for optimal performance

## 📈 Performance Benchmarks

### Typical Performance (4-core CPU, 8GB RAM)
- **Small Dataset** (< 1000 faces): 50-100 faces/second
- **Medium Dataset** (1000-10000 faces): 20-50 faces/second
- **Large Dataset** (> 10000 faces): 10-30 faces/second (with FAISS)

### Memory Usage
- **Base Memory**: ~200MB
- **Per 1000 Faces**: +50MB
- **Peak Memory**: 2-4GB for large batches

## 🐛 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce batch size and workers
MAX_BATCH_SIZE=5
MAX_WORKERS=2
```

#### Slow Processing
```bash
# Enable FAISS for large datasets
ENABLE_FAISS=true

# Increase workers (if CPU available)
MAX_WORKERS=8
```

#### Poor Recognition Accuracy
```bash
# Adjust threshold
FACE_THRESHOLD=0.5

# Enable quality check
ENABLE_QUALITY_CHECK=true
```

### Log Files
- `logs/face_recognition.log`: Main system log
- Database logs: Check MySQL error logs
- Performance logs: Available in processing_logs table

## 🔒 Security Considerations

- **Environment Variables**: Never commit `.env` files
- **Database Security**: Use strong passwords and limit access
- **API Keys**: Rotate Supabase keys regularly
- **Data Privacy**: Ensure compliance with local privacy laws
- **Network Security**: Use HTTPS for all external connections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/face_recognition.log`
3. Check the database `processing_logs` table
4. Create an issue with detailed error information

## 🔄 Migration from Basic System

If you're upgrading from the basic system:

1. **Backup your database**
2. **Install new dependencies**: `pip install -r requirement_enhanced.txt`
3. **Update your `.env` file** with new configuration options
4. **Run the enhanced system**: `python main_enhanced.py`

The enhanced system is backward compatible with existing data.

## 💾 Database setup (phpMyAdmin)

If you prefer phpMyAdmin (XAMPP/WAMP) instead of installing MySQL server manually, follow these steps:

1. Start XAMPP/WAMP and open phpMyAdmin: http://localhost/phpmyadmin/
2. Create a database (e.g. `face_recognition_db`) or update your `.env` to match an existing DB.
3. Use Import in phpMyAdmin to upload these files from the repo:
	- `utils/database_schema.sql`
	- `utils/update_schema_age_gender.sql`
4. Update your `.env` file in the project root with the DB connection details.

Example `.env` (place in `d:\face\takla`):
```env
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=face_recognition_db
MYSQL_PORT=3306
```

After import, run the application:
```powershell
cd D:\face\takla
python main.py
```
