import requests
import json
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path

# law_list.csv는 https://open.law.go.kr/LSO/lab/lawListSupport.do 로부터 다운로드 받음.
law_df = pd.read_csv('./law_list.csv')

law_df['법령명'] = law_df['법령명'].str.replace(r'[^가-힣0-9]', '_', regex=True)

for idx in tqdm(range(law_df.shape[0]), desc='모든 법령 json파일 가져오는 중'):
    row = law_df.iloc[idx]
    id = row['법령ID']
    response = requests.get(f'http://www.law.go.kr/DRF/lawService.do?OC=kevin2001112&target=law&ID={int(id)}&type=JSON')
    if response.status_code == 200:
        data = response.json()
        file_name = row['법령명']
        out = os.path.join(Path(__file__).resolve().parent, "law_json", f"data_{idx}.json")
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        print('API 요청 실패:', response.status_code)