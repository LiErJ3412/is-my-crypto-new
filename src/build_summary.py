# run summarize for all the problems
import requests
import json
from .utils import read_problem, problem_filenames, dump_json_safe, dump_json_safe_utf8
import hashlib
import asyncio
from tqdm.auto import tqdm
import time

start_time = time.time()
with open("settings_sample.json") as f:
    settings = json.load(f)

# 创建自定义API客户端类，替换Together API
class SiliconFlowClient:
    def __init__(self, api_key=None):
        self.api_key = "Bearer "+settings["Bearer_API_KEY"]
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": "Bearer "+settings["Bearer_API_KEY"],
            "Content-Type": "application/json"
        }
    
    async def chat(self):
        return self

    class completions:
        @staticmethod
        async def create(messages, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", max_new_tokens=1024, **kwargs):
            url = "https://api.siliconflow.cn/v1/chat/completions"
            headers = {
                "Authorization": "Bearer "+settings["Bearer_API_KEY"],
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # 使用DeepSeek R1 32B模型
                "stream": False,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.7),
                "top_k": kwargs.get('top_k', 50),
                "frequency_penalty": kwargs.get('frequency_penalty', 0.5),
                "n": 1
            }
            
            # 发送请求
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            
            result = response.json()
            
            class ChoiceMessage:
                def __init__(self, content):
                    self.content = content
            
            class Choice:
                def __init__(self, message):
                    self.message = message
            
            class Usage:
                def __init__(self, total_tokens):
                    self.total_tokens = total_tokens
            
            class Response:
                def __init__(self, choices, usage):
                    self.choices = choices
                    self.usage = usage
            
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            total_tokens = result.get('usage', {}).get('total_tokens', 0)
            
            return Response(
                choices=[Choice(ChoiceMessage(content))],
                usage=Usage(total_tokens)
            )

# 创建客户端实例
client = SiliconFlowClient()


def check_processed(p, template):
    ORIGINAL = p["statement"]
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    for f in p["processed"]:
        if f["prompt_md5"][:8] == prompt_md5:
            return True
    return False


async def process(p, template, delay = 0):
    # sleep for delay first
    await asyncio.sleep(delay)
    ORIGINAL = p["statement"]
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    template_md5 = hashlib.md5(template.encode("utf-8")).hexdigest()[:8]
    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    already_processed = False
    for f in p["processed"]:
        if f["prompt_md5"][:8] == prompt_md5:
            already_processed = True
    if already_processed:
        return
    # print(prompt, prompt_md5)
    # print(num_tokens_from_string(prompt))
    result = None
    try:
        # 使用SiliconFlow API替代Together API
        completion_client = await client.chat()
        response = await completion_client.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                { "role": "assistant", "content": "Simplified statement:" }
            ],
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # 使用DeepSeek V3模型
            max_new_tokens=1024, 
        )
        result = response.choices[0].message.content.strip()
        print(f"Number of tokens spent: {response.usage.total_tokens}")
    except Exception as e:
        print("Error while prompting:", e)
    if result is None:
        return []
    return [
        {
            "prompt_md5": prompt_md5,
            "template_md5": template_md5,
            "result": result,
        }
    ]


async def process_all_problems():
    # apparently some mysterious OJs are spending my money ;_;
    #goodojs = ['UOJ', 'Codeforces', '洛谷', 'DMOJ', 'HDU', 'CodeChef', 'AtCoder', 'LibreOJ', 'TopCoder', 'SPOJ', '51Nod', '黑暗爆炸', 'UVA'] #, 'USACO'
    #badojs = ['HYSBZ', 'BZOJ']
    fns = list(problem_filenames())
    chunk_size = 2
    gap_every = 3
    problem_files = []
    for problem_file_cur in tqdm(fns):#tqdm(range(0,len(fns),chunk_size)):
        try:
            p = read_problem(problem_file_cur)
        except Exception as e:
            print('error',problem_file_cur,e)
            continue
        need_work = False
        for template in settings["TEMPLATES"]:
            if 'processed' in p and check_processed(p, template):
                continue
            need_work = True
        if need_work:
            problem_files.append(problem_file_cur)
        if len(problem_files) >= chunk_size or problem_file_cur == fns[-1]:
            for template in settings["TEMPLATES"]:
                t0 = time.time()
                tasks = []
                notprocessed = []
                for idx, problem_file in enumerate(problem_files):
                    p = read_problem(problem_file)
                    if "processed" not in p:
                        p["processed"] = []
                    if check_processed(p, template):
                        continue
                    notprocessed.append(problem_file)
                    tasks.append(process(p, template, idx * gap_every))
                if not len(tasks):
                    continue
                WAIT = chunk_size * gap_every + .5
                results = await asyncio.gather(*tasks)
                for problem_file, result in zip(notprocessed, results):
                    if not len(result):
                        WAIT = 6
                        continue
                    p = read_problem(problem_file)
                    if "processed" not in p:
                        p["processed"] = []
                    p["processed"].extend(result)
                    print(problem_file)
                    dump_json_safe_utf8(p, problem_file)
                t1 = time.time()
                print('time elapsed',t1-t0)
                # wait till WAIT
                if t1-t0 < WAIT:
                    await asyncio.sleep(WAIT-(t1-t0))
            problem_files = []

if __name__ == "__main__":
    asyncio.run(process_all_problems())