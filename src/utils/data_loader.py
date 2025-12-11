"""
Data Loading Utilities for Spider and WikiSQL datasets
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
import torch


class SpiderDataset(Dataset):
    """Spider dataset for Text-to-SQL"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        tokenizer=None,
        max_length: int = 2048,
        include_schema: bool = True
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_schema = include_schema
        
        # Load data
        self.examples = self.load_spider_data()
        self.db_schemas = self.load_database_schemas()
    
    def load_spider_data(self) -> List[Dict[str, Any]]:
        """Load Spider dataset"""
        file_path = self.data_path / f"{self.split}.json"
        
        if not file_path.exists():
            print(f"Downloading Spider dataset...")
            dataset = load_dataset("spider", split=self.split)
            return list(dataset)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_database_schemas(self) -> Dict[str, str]:
        """Load database schemas"""
        schemas = {}
        tables_file = self.data_path / "tables.json"
        
        if tables_file.exists():
            with open(tables_file, 'r') as f:
                tables_data = json.load(f)
            
            for db in tables_data:
                db_id = db['db_id']
                schema_str = self.format_schema(db)
                schemas[db_id] = schema_str
        
        return schemas
    
    def format_schema(self, db_info: Dict[str, Any]) -> str:
        """Format database schema as text"""
        schema_parts = []
        
        table_names = db_info.get('table_names_original', [])
        column_names = db_info.get('column_names_original', [])
        column_types = db_info.get('column_types', [])
        
        # Group columns by table
        tables = {}
        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:
                continue
            
            table_name = table_names[table_idx]
            if table_name not in tables:
                tables[table_name] = []
            
            col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
            tables[table_name].append(f"{col_name} ({col_type})")
        
        # Format schema
        for table_name, columns in tables.items():
            schema_parts.append(f"Table {table_name}:")
            for col in columns:
                schema_parts.append(f"  - {col}")
        
        return "\n".join(schema_parts)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        question = example.get('question', '')
        sql = example.get('query', example.get('sql', ''))
        db_id = example.get('db_id', '')
        
        schema = self.db_schemas.get(db_id, '') if self.include_schema else ''
        
        # Format prompt
        prompt = self.format_prompt(question, schema)
        
        if self.tokenizer is not None:
            # Tokenize
            encodings = self.tokenizer(
                prompt,
                sql,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze(),
                'question': question,
                'sql': sql,
                'schema': schema,
                'db_id': db_id
            }
        else:
            return {
                'question': question,
                'sql': sql,
                'schema': schema,
                'db_id': db_id,
                'prompt': prompt
            }
    
    def format_prompt(self, question: str, schema: str) -> str:
        """Format question and schema as prompt"""
        if schema:
            prompt = f"""Given the following database schema:

{schema}

Generate a SQL query to answer this question:
Question: {question}

SQL Query: """
        else:
            prompt = f"""Generate a SQL query to answer this question:
Question: {question}

SQL Query: """
        
        return prompt


class WikiSQLDataset(Dataset):
    """WikiSQL dataset for Text-to-SQL"""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_length: int = 2048
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data using Hugging Face datasets
        self.dataset = load_dataset("wikisql", split=split)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        
        question = example['question']
        table = example['table']
        sql = example['sql']
        
        # Format schema from table
        schema = self.format_schema(table)
        
        # Format prompt
        prompt = self.format_prompt(question, schema)
        
        # Convert SQL dict to string
        sql_str = self.sql_dict_to_string(sql, table)
        
        if self.tokenizer is not None:
            encodings = self.tokenizer(
                prompt,
                sql_str,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze(),
                'question': question,
                'sql': sql_str,
                'schema': schema
            }
        else:
            return {
                'question': question,
                'sql': sql_str,
                'schema': schema,
                'prompt': prompt
            }
    
    def format_schema(self, table: Dict[str, Any]) -> str:
        """Format table schema"""
        header = table['header']
        types = table.get('types', ['text'] * len(header))
        
        schema_parts = [f"Table {table.get('name', 'table')}:"]
        for col_name, col_type in zip(header, types):
            schema_parts.append(f"  - {col_name} ({col_type})")
        
        return "\n".join(schema_parts)
    
    def format_prompt(self, question: str, schema: str) -> str:
        """Format prompt"""
        return f"""Given the following database schema:

{schema}

Generate a SQL query to answer this question:
Question: {question}

SQL Query: """
    
    def sql_dict_to_string(self, sql: Dict[str, Any], table: Dict[str, Any]) -> str:
        """Convert WikiSQL dict format to SQL string"""
        # This is a simplified version; you might want to use the official WikiSQL converter
        sel = sql['sel']
        agg = sql['agg']
        conds = sql['conds']
        
        agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']
        
        # SELECT clause
        column = table['header'][sel]
        if agg > 0:
            select_clause = f"SELECT {agg_ops[agg]}({column})"
        else:
            select_clause = f"SELECT {column}"
        
        # FROM clause
        from_clause = f"FROM {table.get('name', 'table')}"
        
        # WHERE clause
        where_parts = []
        for col_idx, op_idx, value in conds:
            col_name = table['header'][col_idx]
            op = cond_ops[op_idx] if op_idx < len(cond_ops) else '='
            where_parts.append(f"{col_name} {op} '{value}'")
        
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)
        else:
            where_clause = ""
        
        # Combine
        sql_str = f"{select_clause} {from_clause}"
        if where_clause:
            sql_str += f" {where_clause}"
        
        return sql_str


class DataLoader:
    """Main data loader class"""
    
    @staticmethod
    def load_spider(data_path: str, split: str = "train", **kwargs) -> SpiderDataset:
        """Load Spider dataset"""
        return SpiderDataset(data_path, split, **kwargs)
    
    @staticmethod
    def load_wikisql(split: str = "train", **kwargs) -> WikiSQLDataset:
        """Load WikiSQL dataset"""
        return WikiSQLDataset(split, **kwargs)
    
    @staticmethod
    def create_dataloaders(
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


# Convenience functions for rStar-SQL
def load_spider_data(data_path: str) -> List[Dict[str, Any]]:
    """Load Spider data as list of dicts (for rStar-SQL)"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def load_wikisql_data(data_path: str) -> List[Dict[str, Any]]:
    """Load WikiSQL data as list of dicts (for rStar-SQL)"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
