"""
Prompt Engineering Utilities
"""

from typing import Dict, List, Optional


class PromptEngineer:
    """Utilities for creating effective Text-to-SQL prompts"""
    
    @staticmethod
    def zero_shot_prompt(question: str, schema: str) -> str:
        """Zero-shot prompting"""
        return f"""Given the following database schema:

{schema}

Generate a SQL query to answer this question:
Question: {question}

SQL Query:"""
    
    @staticmethod
    def few_shot_prompt(
        question: str,
        schema: str,
        examples: List[Dict[str, str]],
        max_examples: int = 3
    ) -> str:
        """Few-shot prompting with examples"""
        prompt_parts = ["Here are some examples of questions and their SQL queries:\n"]
        
        for i, example in enumerate(examples[:max_examples], 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"SQL: {example['sql']}\n")
        
        prompt_parts.append(f"Now, given the following database schema:\n\n{schema}\n")
        prompt_parts.append(f"Generate a SQL query to answer this question:")
        prompt_parts.append(f"Question: {question}\n")
        prompt_parts.append("SQL Query:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def chain_of_thought_prompt(question: str, schema: str) -> str:
        """Chain-of-thought prompting"""
        return f"""Given the following database schema:

{schema}

Let's solve this step by step:

Question: {question}

Step 1: Identify the relevant tables and columns
Step 2: Determine the type of query (SELECT, JOIN, etc.)
Step 3: Construct the WHERE clause if needed
Step 4: Add any necessary aggregations or grouping

SQL Query:"""
    
    @staticmethod
    def schema_linking_prompt(question: str, schema: str) -> str:
        """Prompt with explicit schema linking"""
        return f"""Task: Generate a SQL query to answer the question based on the given schema.

Database Schema:
{schema}

Instructions:
1. Carefully read the question and identify key entities
2. Match entities to table and column names in the schema
3. Construct a valid SQL query

Question: {question}

Let's break this down:
- Entities in question: [Identify entities]
- Relevant tables: [List tables]
- Relevant columns: [List columns]

Final SQL Query:"""
    
    @staticmethod
    def with_hints_prompt(
        question: str,
        schema: str,
        hints: Optional[List[str]] = None
    ) -> str:
        """Prompt with additional hints"""
        prompt_parts = [f"Given the following database schema:\n\n{schema}\n"]
        
        if hints:
            prompt_parts.append("Hints:")
            for hint in hints:
                prompt_parts.append(f"- {hint}")
            prompt_parts.append("")
        
        prompt_parts.append(f"Generate a SQL query to answer this question:")
        prompt_parts.append(f"Question: {question}\n")
        prompt_parts.append("SQL Query:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def self_consistency_prompt(question: str, schema: str, num_paths: int = 3) -> str:
        """Prompt for self-consistency (generate multiple solutions)"""
        return f"""Given the following database schema:

{schema}

Generate {num_paths} different SQL queries that could answer this question:
Question: {question}

SQL Query 1:"""
    
    @staticmethod
    def format_schema_compact(schema_dict: Dict) -> str:
        """Format schema in compact format"""
        parts = []
        for table_name, columns in schema_dict.items():
            col_str = ", ".join([f"{col['name']}({col['type']})" for col in columns])
            parts.append(f"{table_name}({col_str})")
        return " | ".join(parts)
    
    @staticmethod
    def format_schema_detailed(schema_dict: Dict) -> str:
        """Format schema in detailed format"""
        parts = []
        for table_name, info in schema_dict.items():
            parts.append(f"Table: {table_name}")
            if 'description' in info:
                parts.append(f"  Description: {info['description']}")
            parts.append("  Columns:")
            for col in info.get('columns', []):
                col_str = f"    - {col['name']} ({col['type']})"
                if 'description' in col:
                    col_str += f": {col['description']}"
                if col.get('primary_key'):
                    col_str += " [PRIMARY KEY]"
                if col.get('foreign_key'):
                    col_str += f" [FOREIGN KEY -> {col['foreign_key']}]"
                parts.append(col_str)
            parts.append("")
        return "\n".join(parts)
    
    @staticmethod
    def add_sql_guidelines(base_prompt: str) -> str:
        """Add SQL writing guidelines to prompt"""
        guidelines = """
SQL Writing Guidelines:
- Use proper SQL syntax
- Use table aliases for readability
- Use JOIN instead of implicit joins
- Add appropriate WHERE clauses
- Use aggregate functions when needed (COUNT, SUM, AVG, MAX, MIN)
- Use GROUP BY with aggregate functions
- Use ORDER BY for sorting
- Use LIMIT for restricting results

"""
        return guidelines + base_prompt
    
    @staticmethod
    def create_instruction_prompt(question: str, schema: str, instruction: str = None) -> str:
        """Create prompt with custom instruction"""
        default_instruction = "You are an expert SQL query generator. Generate only the SQL query without any explanation or additional text."
        
        instruction = instruction or default_instruction
        
        return f"""{instruction}

Database Schema:
{schema}

Question: {question}

SQL Query:"""
