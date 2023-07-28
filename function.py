import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import pickle


# LLM prompt
get_ans_template = """Use the following pieces of context to answer the question at the end. 
Answer with one paragraph unless specified by the question.
{context}
Question: {question}

Does the context provide information to the question? if yes, give the answer. If no, say "I don't know the answer >.<"
Helpful Answer:"""

ques_gen_template = """
Based on the text below, generate five meaningful questions.
Give the questions is a numeric list form. Seperate the questions with the "@" symbol.
{text}
"""

ans_example = """
NumPy is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more. At the core of the NumPy package is the ndarray object, which encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.
"""

good_ques_example = [
"1. What is the purpose of the NumPy library?",
"2. What is the core of the NumPy package?",
"3. What type of data does the ndarray object encapsulate?",
"4. What operations are performed in compiled code for performance?",
"5. What are some of the features of the NumPy library?"
]

prompt_gen_ques = PromptTemplate(
            input_variables=["text"],
            template=ques_gen_template,
        )

def load_models():
    try:
        openai_api_key = st.secrets["openai_api_key"]
    except:
        with open('OPENAI_API_KEY.pkl', 'rb') as f:
            OPENAI_API_KEY = pickle.load(f) # deserialize using load()
        openai_api_key = OPENAI_API_KEY

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)    
    return llm, openai_embeddings

class Session:

    def __init__(self, llm, embeding_model):
        self.openai_embeddings = embeding_model
        self.llm = llm
        self.qa_chain = None # based on question
        self.session_docs = None
        self.load_successful = True
        self.ques = ""
        self.ans = ""
        self.good_ques = []

    def load_web_data(self, web_address):
        bar = st.progress(0)
        loading_feedback = st.empty()
        loading_feedback.write("loading: " + web_address)
        self.load_successful = False
        # 1) load Document
        #st.write(f"running {np.random.randint(1,10)}")
        loader = WebBaseLoader(web_address)
        data = loader.load()
        if isinstance(data[0].metadata["language"],str) != True:
            data[0].metadata["language"] = "n/a"
        self.session_docs = data

        bar.progress(25)
        # 2) Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        all_splits = text_splitter.split_documents(data)
        bar.progress(50)
        # 3) Store 
        vectorstore = Chroma.from_documents(documents=all_splits,embedding=self.openai_embeddings)
        bar.progress(75)
        # create chain
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=get_ans_template,)
        self.qa_chain = RetrievalQA.from_chain_type(self.llm,
                                            retriever=vectorstore.as_retriever(),
                                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        bar.progress(100)
        loading_feedback.write(f"loaded: {web_address}")
        bar.empty()
        self.load_successful = True
    
    def get_answer(self, question):
        self.question = question
        result = self.qa_chain({"query": question})
        self.ans = result['result']
        return self.ans

    def get_good_ques(self, text):
        final_prompt = prompt_gen_ques.format(text=text)
        good_ques = self.llm(final_prompt)
        good_ques_list = good_ques.split("@")
        formated_ques_list = []
        for i, ques in enumerate(good_ques_list):
            if ques == "":
                break
            formated_ques = "".join(ques.splitlines())
            formated_ques_list.append(formated_ques)

        self.good_ques = formated_ques_list
        return self.good_ques
    
