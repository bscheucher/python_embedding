import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

emb = embeddings.embed_query("I can see a bad moon rising.")

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=200,
#     chunk_overlap=0
# )
# loader = TextLoader("facts.txt")
# docs = loader.load_and_split(text_splitter=text_splitter)
#
# for doc in docs:
#     print(doc.page_content)
#     print("\n")


print(emb)
