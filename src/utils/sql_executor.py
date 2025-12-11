"""
SQL Execution and Validation Utilities
"""

import sqlite3
import timeout_decorator
from typing import List, Tuple, Any, Optional
import re


class SQLExecutor:
    """Execute SQL queries safely against databases"""
    
    def __init__(self, db_path: Optional[str] = None, timeout: int = 10):
        self.db_path = db_path
        self.timeout = timeout
    
    def execute_query(self, sql: str, db_path: Optional[str] = None) -> Tuple[bool, Optional[List], Optional[str]]:
        """
        Execute SQL query and return results
        
        Args:
            sql: SQL query to execute
            db_path: Optional database path (overrides instance db_path)
        
        Returns:
            Tuple of (success, results, error_message)
        """
        path = db_path or self.db_path
        if not path:
            return False, None, "No database path provided"
        
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            
            # Execute with timeout
            cursor.execute(sql)
            results = cursor.fetchall()
            
            conn.close()
            return True, results, None
            
        except sqlite3.Error as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, str(e)
    
    @timeout_decorator.timeout(10)
    def execute_with_timeout(self, sql: str, db_path: Optional[str] = None) -> List:
        """Execute SQL with timeout"""
        path = db_path or self.db_path
        if not path:
            raise ValueError("No database path provided")
        
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax without execution"""
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, None
        except sqlite3.Error as e:
            return False, str(e)
    
    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalize SQL query for comparison"""
        # Convert to uppercase
        sql = sql.upper()
        
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql)
        
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        return sql.strip()
    
    @staticmethod
    def compare_results(results1: List, results2: List) -> bool:
        """Compare two query results"""
        if results1 is None or results2 is None:
            return results1 == results2
        
        # Sort results for comparison
        try:
            sorted1 = sorted([sorted(row) for row in results1])
            sorted2 = sorted([sorted(row) for row in results2])
            return sorted1 == sorted2
        except:
            return results1 == results2
    
    @staticmethod
    def extract_sql_components(sql: str) -> dict:
        """Extract components from SQL query"""
        sql_upper = sql.upper()
        
        components = {
            'select': bool(re.search(r'\bSELECT\b', sql_upper)),
            'from': bool(re.search(r'\bFROM\b', sql_upper)),
            'where': bool(re.search(r'\bWHERE\b', sql_upper)),
            'join': bool(re.search(r'\bJOIN\b', sql_upper)),
            'group_by': bool(re.search(r'\bGROUP BY\b', sql_upper)),
            'having': bool(re.search(r'\bHAVING\b', sql_upper)),
            'order_by': bool(re.search(r'\bORDER BY\b', sql_upper)),
            'limit': bool(re.search(r'\bLIMIT\b', sql_upper)),
            'aggregate': bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\b', sql_upper)),
            'subquery': bool(re.search(r'\(SELECT\b', sql_upper)),
        }
        
        return components
    
    @staticmethod
    def is_valid_sql(sql: str) -> bool:
        """Basic validation of SQL query"""
        sql_upper = sql.upper().strip()
        
        # Must start with a valid SQL keyword
        valid_starts = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        if not any(sql_upper.startswith(keyword) for keyword in valid_starts):
            return False
        
        # Must have balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False
        
        # Must have balanced quotes
        if sql.count("'") % 2 != 0 or sql.count('"') % 2 != 0:
            return False
        
        return True
