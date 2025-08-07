# hyde.py
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load models
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2", max_length=256, do_sample=True)

# Wrap into langchain-compatible pipeline
llm = HuggingFacePipeline(pipeline=generator)

# Hypothetical Answer Prompt
hypo_prompt = PromptTemplate.from_template("Generate a possible answer to the question: {question}")

def generate_hypothetical_answer(question):
    return llm.predict(hypo_prompt.format(question=question))

def create_hyde_retriever(chroma_path="chroma_db"):
    return Chroma(persist_directory=chroma_path, embedding_function=embed_model)

def answer_with_hyde(query, retriever):
    # Step 1: Generate hypothetical answer
    hypothetical = generate_hypothetical_answer(query)
    
    # Step 2: Embed hypothetical and retrieve
    docs = retriever.similarity_search(hypothetical, k=3)

    # Step 3: Use original question + docs to answer
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain.run(query)
