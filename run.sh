#!/bin/bash

# echo "🚀 데이터 다운로드"
# python ./data/corpus/get_criminal_kmmlu.py
# python ./data/corpus/get_law_json.py 

echo "🚀 데이터 구축"

echo "✅ JSON 파싱 -> CSV"
python ./data/corpus/json_to_csv.py

echo "✅ 임베딩벡터 생성 -> FAISS 인덱싱 저장"
python get_index.py

echo "🚀 평가"
python evaluation.py

