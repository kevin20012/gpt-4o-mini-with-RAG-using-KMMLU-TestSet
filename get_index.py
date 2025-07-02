from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Get Criminal Law Corpus
loader = CSVLoader(file_path='./data/corpus/law_corpus.csv')
data = loader.load() 

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

batch_size = 300  # 배치 크기를 적절히 설정
num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

# 배치별로 FAISS에 임베딩 추가
vectorstore = None
for i in tqdm(range(num_batches), desc="Processing documents"):
    batch_data = data[i*batch_size : (i+1)*batch_size]
    processed_batch = []
    for doc in batch_data:
        parts = doc.page_content.split('law: ', 1)  # 최대 1번만 분할
        new_content = parts[1] if len(parts) > 1 else doc.page_content
        
        # 메타데이터 유지하며 새 Document 생성
        new_doc = Document(
            page_content=new_content,
            metadata=doc.metadata
        )
        processed_batch.append(new_doc)
    # print(f"Batch {i} - First doc: {processed_batch[0].page_content[:50]}...")
    
    # 각 배치에 대해 임베딩을 생성하고, FAISS 인덱스에 추가
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents=processed_batch, embedding=embeddings)
    else:
        vectorstore.add_documents(processed_batch)
        
# 벡터 db 저장
vectorstore.save_local('./DB/faiss_law')