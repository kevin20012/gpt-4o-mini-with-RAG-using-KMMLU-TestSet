from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local('./DB/faiss_law', embeddings, allow_dangerous_deserialization=True)

client = OpenAI()

def get_docs(query, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k":top_k})
    docs = retriever.invoke(query)
    docs = "\n\n".join([doc.page_content for doc in docs])
    return docs

def get_prompt(query, top_k=5, method='normal'):
    question = query['question']
    choice = query['choice']
    if method=='normal':
        docs = get_docs(question+" "+choice, top_k=top_k)

    system = "다음 문서를 참고하여 주어진 4가지 보기를 잘 보고 질문에 맞는 답의 숫자를 하나만 고르세요. 가능한 답변: 1,2,3,4\n"
    content = f"문서:\n{docs}\n질문:\n{question}\n보기:\n{choice}\n답변:"

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    
    return prompt

def remove_(answer):
    ''' 불필요한 기호 제거 '''
    if '답변은' in answer:
        answer = answer.split('답변은')[1]
    elif '정답은' in answer:
        answer = answer.split('정답은')[1]
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
        temperature=0.2
    )
    answer = answer.choices[0].message.content.strip()
    return answer

if __name__=='__main__':
    df = pd.read_csv('./data/criminal_kmmlu.csv')
    df_1 = df.iloc[0]
    question = df_1['question']
    choice = f"1: {df_1['A']}\n2: {df_1['B']}\n3: {df_1['C']}\n4: {df_1['D']}"
    answer = df_1['answer']

    query_dict = {
        "question": question,
        "choice": choice
    }

    prompt = get_prompt(query_dict, top_k=10, method='normal')
    # print(prompt)

    print(get_answer(prompt, client))








