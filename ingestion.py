import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

#Description
"""
1. We are chunking the text that is being passed through the LLM using a text splitter because there is a token limit
2. there are many different techniques for chunking and splitting text depending on what you are trying to accomplish
    a. For example, you may want to split up html text; in that case you may split on HTML specific characters to break up the text
"""


load_dotenv()


#Overlapping data explanation
"""
There are some cases where the beginning and end of chunks corresponds to the beginning and end of another truck that together create context
Overlapping helps preserve context from the text
It is used to capture the context between chunks
"""


if __name__ == '__main__':
    print("Ingesting")
    #Loading in document
    loader = TextLoader(r"C:\Users\ifran\OneDrive\Desktop\Dev\GenAI\Udemy LangChain GenAI Course\intro-to-vector-dbs\mediumblog1.txt", encoding='utf-8')
    document = loader.load()

    print("splitting...")
    #Splitting up document into multiple chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    #Passing chunks through neural network to create vector embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

    print("ingesting...")
    #Ingesting vectors in vector store/vector database
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")

    #Augmented step just means that it is taking the question from the user and enhancing it with the relevant context store in the vector database.