import streamlit as st
from langchain.llms import GooglePalm
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

from few_shots import few_shots

from dotenv import load_dotenv
import os


load_dotenv()


def get_few_shot_ans():
    db_user ='root'
    db_password ='root'
    db_host ='localhost'
    db_name ='atliq_tshirts'

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)

    # db_chain = SQLDatabaseChain(llm = llm, database = db, verbose = True)

    api_key = os.getenv('GOOGLE_API_KEY')

    llm = GooglePalm(google_api_key= api_key, temperature=0.7)

    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    to_vectorize = [''.join(i.values()) for i in few_shots]

    vectorestore = FAISS.from_texts(to_vectorize, embeddings, few_shots)

    example_selector = SemanticSimilarityExampleSelector(vectorstore= vectorestore, k=2)

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few = FewShotPromptTemplate(example_selector=example_selector, 
                        example_prompt=example_prompt,
                        suffix=PROMPT_SUFFIX, 
                        prefix=_mysql_prompt, input_variables=['input','table_info','top_k'] 
                        )

    sql_db = SQLDatabaseChain.from_llm(llm, db, few)

    return sql_db





