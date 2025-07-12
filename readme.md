# Legal Agent Using RAG (Retrieval-Augmented Generation)
## 실행방법
```shell
cp .env.example .env
docker build -t agent .
docker run -it -v "$(pwd)":/app agent
```
### 실행 결과
아래 결과가 나온 요청 파일과 결과 파일은 루트 디렉토리에 **batch_requests.jsonl**,**batch_output.jsonl**에 있습니다.
```shell
🚀 데이터 구축
✅ JSON 파싱 -> CSV
json -> csv: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1683/1683 [00:02<00:00, 793.36it/s]
✅ 임베딩벡터 생성 -> FAISS 인덱싱 저장
Processing documents: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 279/279 [31:10<00:00,  6.70s/it]
🚀 평가
🚀 프롬프트 만드는 중...
Make Prompt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:33<00:00,  2.14it/s]
Batch job status: validating
Batch job status: in_progress
...
Batch job status: finalizing
...
Batch job completed.
Accuracy: 0.425
```
https://github.com/daekeun-ml/evaluate-llm-on-korean-dataset/blob/main/DETAILED_RESULTS.md 로부터 KMMLU(criminal_law) 테스테셋에서의 결과를 얻고 비교
||KMMLU(Criminal_law)|
|:--:|:--:|
|0-shot|37|
|5-shot|37.5|
|with RAG (ours)|**42.5**|

## 코드설명
### 데이터 전처리 및 구축
데이터의 리스트(data/corpus/law_list.csv)는 국가법령정보 공동활용 사이트 https://open.law.go.kr/LSO/lab/lawListSupport.do 페이지에서 법령 부분만 다운로드 받았습니다.

#### 🔥 데이터 다운로드
```shell
######################################################## 
# 아래 코드는 데이터를 다운로드 받는 코드입니다. 하지만 소요시간 측정시 이미 데이터가 다운로드 되어있다고 가정하여 총 소요 시간에서 제외하였습니다.
######################################################## 

# HAERAE-HUB/KMMLU 로부터 criminal_law 부분만의 test set을 다운로드 받는 코드입니다.
python get_criminal_kmmlu.py
# law_list.csv 로부터 모든 법령을 api를 사용하여 json파일로 다운로드 받는 코드입니다.
python get_law_json.py 
```

#### 🔥 데이터 구축
```shell
########################################################
# 아래 코드부터 총 소요 시간에 포함됩니다.
######################################################## 
# 🚀 json 파싱
# 다운로드 받은 법령 json 파일로부터 조문 내용을 파싱하기 위한 코드입니다.
# 조문 단위 내의 조문 내용, 항, 호를 알맞게 처리하고, 각 조문 내용 앞에는 법령 이름을 모두 붙여주었습니다.
# 즉, "법령이름:\n제1조 ~~ \n1. ~~ \n2. ~~ ..." 과 같이 csv의 각 행에 저장되도록하였습니다.
# 각 법령에 해당하는 csv는 data/corpus/law_csv에 저장되며, 모든 법령의 내용을 합친 csv는 data/corpus/law_corpus.csv로 저장됩니다.
python json_to_csv.py

# 🚀 인덱싱
# 만들어진 law_corpus.csv 파일로부터 text-embedding-3-small 임베딩모델을 사용하여 각 문자를 임베딩벡터로 변환한 뒤, 각 벡터들을 효율적으로 인덱싱하기 위해 FAISS를 사용합니다.
# 만들어진 인덱싱파일은 DB/faiss_law에 저장됩니다.
python get_index.py
```

### RAG를 사용하여 평가하기
#### 🔥 한개의 쿼리에 대해 답변 얻기
agent.py 에는 LLM으로부터 RAG를 사용하여 답변을 얻기 위한 4가지 함수가 정의되어있습니다.  
또한 실행 시 예시로 첫번째 테스트셋에 대한 답변을 얻을 수 있습니다.
- **get_docs** : FAISS 인덱싱을 사용하여 데이터셋의 질문과 보기를 바탕으로 코사인 유사도 기준 가장 가까운 top_k개의 문서만을 가져옵니다.
- **get_prompt** : KMMLU데이터셋의 질문과 보기를 사용하여, **get_docs**로 top_k개의 유사한 문서를 가져옵니다. 얻어온 문서는 질문과 보기와 함께 아래와 같은 프롬프트로 만든 뒤 반환합니다.  
    - **top_k**: 상위 몇개의 문서를 가져올지 결정
    - **method**: **normal** -> 기본 retriever, **multi-query** -> multi-query retriever 사용
    ```
    system = "다음 문서를 참고하여 주어진 4가지 보기를 잘 보고 질문에 맞는 답의 숫자를 하나만 고르세요. 가능한 답변: 1,2,3,4\n"
    content = f"문서:\n{docs}\n질문:\n{question}\n보기:\n{choice}\n답변:"

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    ```
- **get_answer** : gpt-4o-mini를 사용하여 **get_prompt**로부터 얻은 프롬프트를 사용해 답변을 얻습니다.
- **remove_** : **get_answer**로부터 얻어낸 답변내에 보기 중 고른 숫자만을 파싱합니다. 답변 분석 결과, "답변은~", "정답은~" 뒤에 답이 오는 경우가 많아 이를 기준으로 뒤에서 가장 첫번째 숫자를 정답 숫자로 만들었습니다. 만약 "답변은~", "정답은~" 말이 없는 경우, 답변에서 가장 첫번째로 나온 숫자를 답으로 합니다.
```shell
python agent.py
```
#### 🔥 Open AI Batch API를 사용해서 테스트 데이터 전체에 대한 결과 얻기
evaluation.py는 agent.py에서 정의된 함수를 바탕으로 KMMLU-criminal_law 테스트 셋 전체에 대한 결과를 Batch API를 이용해 답변을 얻습니다.  
실행시 루트 디렉토리에 **batch_requests.jsonl**(요청파일), **batch_output.jsonl**(결과파일) 파일을 얻을 수 있습니다.  
**Batch API의 토큰 한계 때문에 top_k: 12로 설정하였으며, 프롬프트 생성 속도 때문에 method: normal로 설정하였습니다.**
```shell
python evalution.py
```