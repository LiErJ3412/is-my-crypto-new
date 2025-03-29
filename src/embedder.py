import pickle
import requests
import json
from .utils import read_problem, problem_filenames, dump_json_safe, dump_json_safe_utf8, dump_pickle_safe
import hashlib
import asyncio
from tqdm.auto import tqdm
import time, os
import random

# 创建自定义API客户端类替代voyageai
class SiliconFlowClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed(self, texts, model=None, input_type=None):
        """使用SiliconFlow API计算文本嵌入"""
        if not texts:
            return type('obj', (object,), {
                'embeddings': [],
                'total_tokens': 0
            })
        
        # 使用文本嵌入API
        payload = {
            "input": texts,
            "encoding_format": "float",
            "model": "BAAI/bge-large-en-v1.5"  # 使用中文模型
        }
        
        response = requests.post(self.url, json=payload, headers=self.headers)
        
        if response.status_code != 200:
            print(f"嵌入API错误: {response.status_code}, {response.text}")
            # 返回空嵌入以防止程序崩溃
            return type('obj', (object,), {
                'embeddings': [[] for _ in texts],
                'total_tokens': 0
            })
        
        result = response.json()
        
        # 解析SiliconFlow API的响应结构
        embeddings = []
        for item in result.get('data', []):
            if 'embedding' in item:
                embeddings.append(item['embedding'])
        
        # 模拟Voyage响应结构
        return type('obj', (object,), {
            'embeddings': embeddings,
            'total_tokens': result.get('usage', {}).get('total_tokens', 0)
        })


with open("settings_sample.json") as f:
    settings = json.load(f)

# 使用SiliconFlowClient替换voyageai客户端
voyage_client = SiliconFlowClient(
    api_key=settings.get('SILICON_API_KEY', "Bearer_API_KEY"),
)

# client = Together(
#     api_key=settings['TOGETHER_API_KEY'],
# )


def processed_promptmd5(statement, template):
    ORIGINAL = statement
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]

import numpy as np

def problem_embeds(problem_file_cur):
    try:
        problem = read_problem(problem_file_cur)
        # load from corresponding npy
    except Exception as e:
        print('error',problem_file_cur,e)
        return None, None
    try:
        embeds = []
        with open(problem_file_cur.replace(".json", ".vopkl"), "rb") as f:
            embeds = pickle.load(f)
    except:
        pass
    return problem, embeds

# quick and dirty vector database implementation
class VectorDB:
    def __init__(self):
        pass
    
    def load_all(self, shuffle = False, load_around = None, record_tasks=False, skipped_sources = []):
        self.arr = []
        self.metadata = []
        self.todos = []
        self.sources = {}
        fns = list(problem_filenames())
        if shuffle:
            random.shuffle(fns)
        for problem_file_cur in tqdm(fns):
            # if '洛谷' not in problem_file_cur:
            #     continue
            if load_around is not None and len(self.arr) > load_around * 2:
                break
            if not record_tasks and not os.path.exists(problem_file_cur.replace(".json", ".vopkl")):
                continue
            problem, embeds = problem_embeds(problem_file_cur)
            if problem is None:
                continue
            statement = problem['statement']
            source = problem['source']
            if source in skipped_sources:
                continue
            self.sources[source] = self.sources.get(source, 0) + 1
            need_work = False
            for template in settings["TEMPLATES"]:
                md5 = processed_promptmd5(statement, template)
                found = False
                for m, u in embeds:
                    if m[:8] == md5:
                        found = True
                        self.arr.append(np.array(u/np.linalg.norm(u),dtype=np.float16))
                        self.metadata.append((problem_file_cur, source, len(statement.strip())))
                        break
                if not found:
                    need_work = True
            if need_work and record_tasks:
                self.todos.append(problem_file_cur)
        print('found',len(self.arr),'embeds')
        self.arr = np.array(self.arr,dtype=np.float16)
        if record_tasks:
            print('found',len(self.todos),'todos')

    
    def complete_todos(self, chunk_size = 32, length_limit = 10000, shuffle = False, max_tokens = 512):
        todos = self.todos
        if shuffle:
            import random
            random.shuffle(todos)
        for i in tqdm(range(0,len(todos),chunk_size)):
            problems = todos[i:i+chunk_size]
            infos = {}
            for problem_file_cur in problems:
                try:
                    full_problem = read_problem(problem_file_cur)
                    statement = full_problem['statement']
                    # load from corresponding npy
                except Exception as e:
                    print('error',problem_file_cur,e)
                    continue
                try:
                    embeds = []
                    with open(problem_file_cur.replace(".json", ".vopkl"), "rb") as f:
                        embeds = pickle.load(f)
                except:
                    pass
                infos[problem_file_cur] = full_problem.get('processed',[]), statement, embeds
            for template in settings["TEMPLATES"]:
                queues = []
                max_length = 0
                for problem_file_cur, (processed, statement, embeds) in infos.items():
                    md5 = processed_promptmd5(statement, template)
                    if any(m[:8] == md5 for m, u in embeds): continue
                    # get processed
                    processed_text = None
                    for f in processed:
                        if f["prompt_md5"][:8] == md5:
                            if len(f['result']) > length_limit:
                                continue # too long?
                            processed_text = f["result"]
                            max_length = max(max_length, len(processed_text))
                    if processed_text is None:
                        continue
                    
                    # 截断文本以适应token限制
                    # 一个粗略估计：中文每个字约1-2个token，英文每4个字符约1个token
                    # 保守估计取1000字符长度(约500 tokens)
                    if len(processed_text) > 1000:
                        print(f"截断文本从 {len(processed_text)} 字符到 1000 字符")
                        processed_text = processed_text[:1000]
                    
                    queues.append((processed_text, problem_file_cur, md5))
                
                if len(queues) == 0:
                    continue
                
                print('batch',len(queues),' maxlen',max_length)
                
                # 将大批量拆分为符合API限制的小批量
                max_batch_size = 32  # SiliconFlow API 的最大批量大小
                
                for j in range(0, len(queues), max_batch_size):
                    batch_queues = queues[j:j+max_batch_size]
                    batch_texts = [x[0] for x in batch_queues]
                    
                    print(f"处理批次 {j//max_batch_size + 1}/{(len(queues) + max_batch_size - 1)//max_batch_size}, 大小: {len(batch_texts)}")
                    
                    try:
                        t0 = time.time()
                        response = voyage_client.embed(
                            batch_texts,
                            model="BAAI/bge-large-en-v1.5",
                            input_type='document'
                        )
                        print('Token spent',response.total_tokens)
                        t1 = time.time()
                        # wait till 0.5s
                        if t1 - t0 < 0.2:
                            time.sleep(0.2 - (t1 - t0))
                        
                        for q, e in zip(batch_queues, response.embeddings):
                            # 检查是否为空嵌入
                            if len(e) == 0:
                                print(f"警告: {q[1]} 的嵌入向量为空")
                                continue
                            infos[q[1]][2].append((q[2], np.array(e)))
                    except Exception as e:
                        print('error',e)
                    
                    # 为避免API速率限制，在批次之间添加短暂延迟
                    time.sleep(0.5)
            
            for problem_file_cur, (processed, statement, embeds) in infos.items():
                if len(embeds) > 0:  # 只保存有嵌入的文件
                    dump_pickle_safe(embeds, problem_file_cur.replace(".json", ".vopkl"))


    def query_nearest(self, emb, k=1000, dedup=True):
        # return the k nearest embeddings with cosine similarity
        # return a list of (cosine similarity, metadata) tuples
        # the list is sorted by cosine similarity
        # normailze emb
        emb = np.array(emb)
        if len(emb.shape) == 1:
            emb = emb[None, :]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        emb = np.array(emb, dtype=np.float16)
        sims = np.max(self.arr @ emb.T, axis=1)
        sims = np.clip((sims+1)/2, 0, 1)  # [-1,1] -> [0,1]
        topk = np.argsort(sims)[::-1]
        nearest = []
        keys = set()
        # print(f'query nearest {len(emb)=} {len(sims)=} {len(topk)=} {k=}')
        for i in topk:
            if dedup:
                key = self.metadata[i][0]
                if key in keys:
                    continue
                keys.add(key)
            nearest.append((sims[i], i))
            if len(nearest) >= k:
                break
        return nearest

if __name__ == "__main__":
    db = VectorDB()
    db.load_all(record_tasks=True)
    db.complete_todos()
