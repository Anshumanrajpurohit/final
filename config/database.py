import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

class MySQLConfig:
    def __init__(self):
        self.host = os.getenv('MYSQL_HOST', 'localhost')
        self.user = os.getenv('MYSQL_USER')
        self.password = os.getenv('MYSQL_PASSWORD')
        self.database = os.getenv('MYSQL_DATABASE')
        self.port = int(os.getenv('MYSQL_PORT', 3306))
    
    def get_connection(self):
        """Get MySQL connection with robust error handling"""
        connection_configs = []
        
        # Primary config with password
        if self.password and self.password.strip():
            connection_configs.append({
                'host': self.host,
                'user': self.user,
                'password': self.password,
                'database': self.database,
                'port': self.port,
                'autocommit': True,
                'auth_plugin': 'mysql_native_password'
            })
        
        # Fallback config without password (for root with no password)
        connection_configs.append({
            'host': self.host,
            'user': self.user,
            'database': self.database,
            'port': self.port,
            'autocommit': True,
            'auth_plugin': 'mysql_native_password'
        })
        
        # Another fallback without auth_plugin
        if self.password and self.password.strip():
            connection_configs.append({
                'host': self.host,
                'user': self.user,
                'password': self.password,
                'database': self.database,
                'port': self.port,
                'autocommit': True
            })
        
        connection_configs.append({
            'host': self.host,
            'user': self.user,
            'database': self.database,
            'port': self.port,
            'autocommit': True
        })
        
        # Try each configuration
        for config in connection_configs:
            try:
                connection = mysql.connector.connect(**config)
                if connection.is_connected():
                    # Ensure autocommit is actually enabled on the connection object
                    try:
                        connection.autocommit = True
                    except Exception:
                        # Older connector versions may not allow setting this attribute; ignore safely
                        pass

                    # Minimal, non-sensitive debug info to confirm which DB we're connected to
                    host = config.get('host')
                    port = config.get('port')
                    database = config.get('database')
                    user = config.get('user')
                    print(f"[DB] Connected to MySQL @ {host}:{port} database={database} user={user}")
                    return connection
            except Error as e:
                continue
        
        # If all fail, try connecting without database to check if credentials work
        try:
            base_config = {
                'host': self.host,
                'user': self.user,
                'port': self.port,
                'autocommit': True
            }
            if self.password and self.password.strip():
                base_config['password'] = self.password
                
            connection = mysql.connector.connect(**base_config)
            if connection.is_connected():
                print(f"✅ Connected to MySQL server, but database '{self.database}' may not exist")
                print(f"💡 Create database with: CREATE DATABASE {self.database};")
                connection.close()
        except Error as e:
            print(f"❌ MySQL connection failed: {e}")
            print(f"🔍 Check credentials - Host: {self.host}, User: {self.user}, Database: {self.database}")
        
        return None