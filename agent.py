from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local('./DB/faiss_law', embeddings, allow_dangerous_deserialization=True)

client = OpenAI()

def get_docs(query, top_k=5, method="normal"):
    retriever = vectorstore.as_retriever(search_kwargs={"k":top_k})
    if method == "normal":
        pass
    elif method == "multi-query":
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0,
            max_tokens=500,
        )
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )
    docs = retriever.invoke(query)
    docs = "\n\n".join([doc.page_content for doc in docs])
    return docs

def get_prompt(query, top_k=5, method='normal'):
    question = query['question']
    choice = query['choice']
    docs = get_docs(question+" "+choice, top_k=top_k, method=method)

    system = "당신은 법률전문가입니다. 주어진 법률 문서를 참고하여 질문과 함께 주어진 보기를 자세히 읽고 생각하여 질문의 정답을 고르세요. 최종 답변은 반드시 1,2,3,4 중 하나의 숫자로 제시하세요. "
    content = f"\n문서:\n{docs}\n질문:\n{question}\n보기:\n{choice}\n답변:"

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    
    return prompt

def remove_(answer):
    ''' 불필요한 기호 제거 '''
    if '답변' in answer:
        answer = answer.split('답변')[1]
    elif '정답' in answer:
        answer = answer.split('정답')[1]
    else:
        pass
    pattern = re.compile(r"(\d+)")
    try:
        answer = re.findall(pattern, answer)[0]
    except:
        answer = -1
    return answer

def get_answer(prompt, client):
    answer = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=prompt,
        temperature=0.0
    )
    answer = answer.choices[0].message.content.strip()
    return answer

if __name__=='__main__':
    df = pd.read_csv('./data/criminal_kmmlu.csv')
    df_1 = df.iloc[6]
    question = df_1['question']
    choice = f"1: {df_1['A']}\n2: {df_1['B']}\n3: {df_1['C']}\n4: {df_1['D']}"
    answer = df_1['answer']

    query_dict = {
        "question": question,
        "choice": choice
    }

    prompt = get_prompt(query_dict, top_k=10, method='normal')
    print(prompt)
    answer = get_answer(prompt, client)
    print('원래 답변:', answer)
    print('정답 파싱:', remove_(answer))








