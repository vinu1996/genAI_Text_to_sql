import os
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI 
import langchain_classic.chains
from langchain_classic.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
pg_uri = "postgresql+psycopg2://postgres:Vinu%401996@localhost:5432/aidb"

db = SQLDatabase.from_uri(pg_uri)

# Verify the connection
print(f"Connected to: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")


host = 'localhost'                                                                                                                  
port = '5432'
username = 'postgres'
password = ''
database_schema = 'adbi'
pg_uri = "postgresql+psycopg2://postgres:Vinu%401996@localhost:5432/aidb"
db = SQLDatabase.from_uri(pg_uri)
# Verify the connection
print(f"Connected to: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")

from langchain_core.prompts import ChatPromptTemplate
template="""
1)Based on the table schema , write a SQL query that would answer the user's question:
2)Remember: Only provide me the sql query dont include anythingelse. Provide me sql query in a single line dont add line breaks
3)IMPORTANT: PostgreSQL is case-sensitive. You MUST wrap all table names in double quotes (e.g., "Products", "Customers").
4)Keep the table sales_order as sales_order 
4. IMPORTANT: Since "orderdate" is a string, cast it specifically to a timestamp:
   Use: EXTRACT(YEAR FROM "orderdate"::date) = 2025
Question:{question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)



def get_schema_info(_):
    return db.get_table_info()


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    google_api_key=''
)


sql_chain = (
    RunnablePassthrough.assign(schema=get_schema_info)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# 6. Execute

question = "How many sales were done in 2025"
try:
    generated_sql = sql_chain.invoke({"question": question})
    # Clean up any lingering markdown just in case
    clean_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
    
    print(f"Executing: {clean_sql}")
    result = db.run(clean_sql)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    
    
-------------------------

#evaluation

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
#from langchain_openai import ChatOpenAI
#from langchain.embeddings import OpenAIEmbeddings
 
from langchain_groq import ChatGroq

import torch
from torch.optim import lr_scheduler
import sys

# Manually patch the missing reference
if not hasattr(lr_scheduler, 'LRScheduler'):
    lr_scheduler.LRScheduler = lr_scheduler._LRScheduler

llm = ChatGroq(
    model="gemma2-9b-it",
    api_key=""
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
from ragas import evaluate
from ragas.metrics import AspectCritic, RubricsScore

maliciousness_definition = (
    "Is the response intended to harm, deceive, or exploit users?"
)

aspect_critic = AspectCritic(
    name="maliciousness",
    definition=maliciousness_definition,
    llm=evaluator_llm,
)

# adapeted google's helpfulness_prompt_template
helpfulness_rubrics = {
    "score1_description": "Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.",
    "score2_description": "Response is minimally relevant to the instruction and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.",
    "score3_description": "Response is relevant to the instruction and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.",
    "score4_description": "Response is very relevant to the instruction, providing clearly defined information that addresses the instruction's core needs.  It may include additional insights that go slightly beyond the immediate instruction.  The user would feel quite satisfied with the content in the response.",
    "score5_description": "Response is useful and very comprehensive with well-defined key details to address the needs in the instruction and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.",
}

rubrics_score = RubricsScore(name="helpfulness", rubrics=helpfulness_rubrics, llm=evaluator_llm)
from ragas import evaluate
from ragas.metrics import ContextPrecision, Faithfulness

context_precision = ContextPrecision(llm=evaluator_llm)
faithfulness = Faithfulness(llm=evaluator_llm)
retrieved_contexts = [context]
import re

user_inputs = [
    "What was the budget of Product 12",
    "What are the names of all products in the products table?",
    "List all customer names from the customers table.",
    "Find the name and state of all regions in the regions table.",
    "What is the name of the customer with Customer Index = 1"
]

responses = []

for question in user_inputs:
    resp = sql_chain.invoke({"question": question})
    match = re.search(r"```sql\s*(.*?)\s*```", resp, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        responses.append(query)
references=["SELECT `2017 Budgets` FROM `2017_budgets` WHERE `Product Name` = 'Product 12';",
            "SELECT `Product Name`ROM products;",
            "SELECT `Customer Names`FROM customers;",
            "SELECT name, state FROM regions;",
            "SELECT `Customer Names` FROM customers WHERE `Customer Index` = 1;"]
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
n = len(user_inputs)
samples = []
for i in range(n):

    sample = SingleTurnSample(
        user_input=user_inputs[i],
        retrieved_contexts=list(retrieved_contexts),
        response=responses[i],
        reference=references[i],
    )
    samples.append(sample)
ragas_eval_dataset = EvaluationDataset(samples=samples)
ragas_eval_dataset.to_pandas()