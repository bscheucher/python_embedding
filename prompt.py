from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
from langchain.globals import set_verbose
from dotenv import load_dotenv

set_verbose(True)

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.invoke("What is an interesting fact about the English language?")

print(result)
