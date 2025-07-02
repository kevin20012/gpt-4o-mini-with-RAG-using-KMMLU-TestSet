import json
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import re

cur_loc = Path(__file__).resolve().parent
law_csv_dir = os.path.join(cur_loc, 'law_csv')
law_json_dir = os.path.join(cur_loc, 'law_json')

def save_csv(file_dict, file_name, save_dir):
    law_name = file_dict['법령']['기본정보']['법령명_한글']
    contents_list = []
    contents = file_dict['법령']['조문']['조문단위']

    if type(contents) != list:
        return

    for item in contents:
        if item['조문여부'] != '조문':
            continue
        else:
            if '항' not in item.keys():
                if type(item['조문내용']) == list:
                    contents_list.append(law_name+":\n"+'\n'.join(item['조문내용'][0]))
                else:
                    contents_list.append(law_name+":\n"+item['조문내용'])
            else:
                hangs = item['항']
                if type(item['조문내용']) == list:
                    result = law_name+":\n"+'\n'.join(item['조문내용'][0])
                else:
                    result = law_name+":\n"+item['조문내용']
                if type(hangs) == list:
                    for hang in hangs:
                        hang_contents = hang['항내용']
                        if type(hang_contents) == list:
                            hang_contents = '\n'.join(hang_contents[0][0])
                        result+=f"\n{hang_contents}"
                        if '호' in hang.keys():
                            hos = hang['호']
                            if type(hos) == list:
                                for ho in hos:
                                    ho_contents = ho['호내용']
                                    result+=f"\n{ho_contents}"
                            else:
                                ho_contents = '\n'.join(ho['호내용'][0][0])
                                result+=f"\n{ho_contents}"
                else:
                    if '호' not in hangs.keys():
                        hang_contents = hang['항내용']
                        result+=f"\n{hang_contents}"
                    else:
                        hos = hangs['호']
                        for ho in hos:
                            ho_contents = ho['호내용']
                            result+=f"\n{ho_contents}"
                contents_list.append(result)


    df = pd.DataFrame({file_name: contents_list})
    df.to_csv(os.path.join(cur_loc, save_dir, f'{file_name}.csv'), index=False, header=False)

for file in tqdm(os.listdir(law_json_dir), desc='json -> csv'):
    file_path = os.path.join(law_json_dir, file)
    file_name = file.split('.')[0]
    with open(file_path, 'r', encoding='utf-8') as f:
        file_dict = json.load(f)
    save_csv(file_dict, file_name, law_csv_dir)

# 하나의 CSV로 만들기
csv_files = os.listdir(law_csv_dir)
df_list = [pd.read_csv(os.path.join(law_csv_dir, f), header=None) for f in csv_files]
merged_df = pd.concat(df_list).rename({0:'law'},axis=1)
merged_df.to_csv(os.path.join(cur_loc,'law_corpus.csv'))