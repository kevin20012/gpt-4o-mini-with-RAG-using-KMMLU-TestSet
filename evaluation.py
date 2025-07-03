from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
import json
import pandas as pd
import time
from agent import get_docs, get_prompt, remove_
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local('./DB/faiss_law', embeddings, allow_dangerous_deserialization=True)

client = openai.OpenAI()

df_test = pd.read_csv('./data/criminal_kmmlu.csv')

#여러 Query 모아 Task 만들기
tasks = []
answers_list = []

print('🚀 프롬프트 만드는 중...')
for i, row in tqdm(enumerate(df_test.itertuples()), desc="Make Prompt", total=len(df_test)):
    question = row.question
    choice = f"1: {row.A}\n2: {row.B}\n3: {row.C}\n4: {row.D}"

    query_dict = {
        "question": question,
        "choice": choice
    }
    prompt = get_prompt(query_dict, top_k=12, method='normal')
    task = {
        "custom_id": f"task_{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": prompt
        }
    }
    tasks.append(task)

with open("batch_requests.jsonl", "w") as f:
    for task in tasks:
        f.write(json.dumps(task) + "\n")

# jsonl 업로드하기
with open("batch_requests.jsonl", "rb") as f:
    response = client.files.create(file=f, purpose="batch")
file_id = response.id

# 배치 작업 생성하기
batch_job = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

# 배치 작업 기다리기
while True:
    batch_job = client.batches.retrieve(batch_job.id)
    status = batch_job.status
    if status == "completed":
        print("Batch job completed.")
        output_file_id = batch_job.output_file_id
        break
    elif status == "failed":
        print("Batch job failed.")
        break
    else:
        print(f"Batch job status: {status}")
    time.sleep(10)
output_file = client.files.content(output_file_id)
output_content = output_file.content.decode('utf-8')
with open("batch_output.jsonl", "w") as f:
    f.write(output_content)

# 완료된 결과 파싱해서 Accuracy 측정하기
df_test = pd.read_csv('./data/criminal_kmmlu.csv')
answers_list = df_test['answer'].to_list()

result_dict_list = []
with open("batch_output.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        result_dict_list.append(data)

answer_pred_list = []
for result in result_dict_list:
    pred = result['response']['body']['choices'][0]['message']['content']
    pred = int(remove_(pred))
    answer_pred_list.append(pred)

score = accuracy_score(answer_pred_list, answers_list)
print('Accuracy:', score)







