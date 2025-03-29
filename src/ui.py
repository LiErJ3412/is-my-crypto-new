import numpy as np
from .embedder import VectorDB, processed_promptmd5
from .utils import read_problem
from tqdm.auto import tqdm
import gradio as gr
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import urllib
import time
import requests
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)

db = VectorDB()
db.load_all()
print("read", len(set(x[0] for x in db.metadata)), "problems")
print(db.metadata[:100])

with open("settings_sample.json") as f:
    settings = json.load(f)

# 创建自定义API客户端类
class SiliconFlowClient:
    def __init__(self, api_key):
        self.api_key = "Bearer "+settings["Bearer_API_KEY"]
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": "Bearer "+settings["Bearer_API_KEY"],
            "Content-Type": "application/json"
        }
    
    async def chat_completion(self, messages, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", temperature=0.7):
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": temperature,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1
        }
        # 使用异步请求
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(self.url, json=payload, headers=self.headers)
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        
        result = response.json()
        # 模拟OpenAI响应结构
        return type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    })
                })
            ]
        })

    # 添加嵌入API方法，替代Voyage API
    # 替换SiliconFlowClient的embed方法
    async def embed(self, texts, model=None, input_type=None):
        """使用SiliconFlow API计算文本嵌入"""
        if not texts:
            return type('obj', (object,), {
                'embeddings': [],
                'total_tokens': 0
            })
        
        # 使用文本嵌入API - 按照正确格式构建请求
        payload = {
            "input": texts,
            "encoding_format": "float",
            "model": "BAAI/bge-large-en-v1.5"  
        }
        
        url = "https://api.siliconflow.cn/v1/embeddings"
        headers = {
            "Authorization": "Bearer "+settings["Bearer_API_KEY"],
            "Content-Type": "application/json"
        }
        
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, headers=headers)
            )
            
            if response.status_code != 200:
                print(f"嵌入API错误: {response.status_code}, {response.text}")
                # 返回空嵌入以防止程序崩溃
                return type('obj', (object,), {
                    'embeddings': [np.random.randn(1024).tolist() for _ in texts],  # 使用随机嵌入
                    'total_tokens': 0
                })
            
            result = response.json()
            print("嵌入API响应:", result.keys())
            
            # 正确解析SiliconFlow API的响应结构
            embeddings = []
            for item in result.get('data', []):
                if 'embedding' in item:
                    embeddings.append(item['embedding'])
            
            # 模拟Voyage响应结构
            return type('obj', (object,), {
                'embeddings': embeddings,
                'total_tokens': result.get('usage', {}).get('total_tokens', 0)
            })
        
        except Exception as e:
            print(f"嵌入API请求异常: {str(e)}")
            # 返回空嵌入以防止程序崩溃
            return type('obj', (object,), {
                'embeddings': [np.random.randn(1024).tolist() for _ in texts],  # 使用随机嵌入
                'total_tokens': 0
            })

# 使用单一客户端替换所有API客户端
silicon_client = SiliconFlowClient(api_key=settings.get('SILICON_API_KEY', ''))

async def querier(statement, *template_choices):
    assert len(template_choices) % 3 == 0
    yields = []
    ORIGINAL = statement.strip()
    t1 = time.time()

    async def process_template(engine, prompt, prefix):
        if 'origin' in engine.lower() or '保' in engine.lower():
            return ORIGINAL
        if 'none' in engine.lower() or '跳' in engine.lower():
            return ''

        prompt = prompt.replace("[[ORIGINAL]]", ORIGINAL).strip()
        
        # 所有LLM请求都通过SiliconFlow客户端处理
        if "deepseek" in engine.lower() or "gemma" in engine.lower() or "gpt" in engine.lower():
            response = await silicon_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant.answer my questions in English."}, 
                    {"role": "user", "content": prompt},
                    # 对于需要前缀的模型，添加助手前缀
                    {"role": "assistant", "content": prefix} if any(m in engine.lower() for m in ["gemma", "deepseek"]) else None
                ],
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            )
            result = response.choices[0].message.content.strip()
            # 对于GPT模型，移除前缀
            if "gpt" in engine.lower():
                result = result.replace(prefix.strip(), '', 1).strip()
            return result
        else:
            raise NotImplementedError(engine)

    tasks = [process_template(template_choices[i], template_choices[i+1], template_choices[i+2]) 
             for i in range(0, len(template_choices), 3)]
    yields = await asyncio.gather(*tasks)

    t2 = time.time()
    print('query llm', t2-t1)
    
    # 使用嵌入API
    response = await silicon_client.embed(
        list(set(y.strip() for y in yields if len(y))),
        #model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        model="BAAI/bge-large-en-v1.5"
    )
    
    print('Token spent', response.total_tokens)
    emb = [d for d in response.embeddings]
    t3 = time.time()
    print('query emb', t3-t2)

    loop = asyncio.get_running_loop()
    nearest = await loop.run_in_executor(executor, db.query_nearest, emb, 5000)
    t4 = time.time()
    print('query nearest', t4-t3)

    sim = np.array([x[0] for x in nearest])
    ids = np.array([x[1] for x in nearest], dtype=np.int32)

    info = '已查找到前' + str(len(sim)) + '个匹配！进入下一页查看结果~'

    return [info, (sim, ids)] + yields



def format_problem(uid, sim):
    # 获取问题数据
    uid = db.metadata[int(uid)][0]
    problem = read_problem(uid)
    statement = problem["statement"].replace("\n", "\n\n")
    title = problem['title']
    
    # 创建HTML展示
    html = f'<p><span style="font-size:22px; font-weight: 500;">{title}</span>&nbsp;&nbsp;<span style="font-size:15px">({round(sim*100)}%)</span></p>\n'
    url = problem["url"]
    problemlink = uid.replace('/',' ').replace('\\',' ').strip().replace('problems vjudge','',1).strip().replace('_','-')
    assert problemlink.endswith('.json')
    problemlink = problemlink[:-5].strip()
    
    # 搜索链接
    link1 = 'https://www.google.com/search?'+urllib.parse.urlencode({'q': problem['source']+' '+title})
    link1_bd = 'https://www.baidu.com/s?'+urllib.parse.urlencode({'wd': problem['source']+' '+title})
    
    html += f'&nbsp;&nbsp;&nbsp;<a href="{url}" target="_blank">VJudge</a>&nbsp;&nbsp;<a href="{link1}" target="_blank">谷歌</a>&nbsp;&nbsp;<a href="{link1_bd}" target="_blank">百度</a>'
    
    # 处理题目简要
    markdown = ''
    rsts = []
    for template in settings.get('TEMPLATES', settings['TEMPLATES']):
        md5 = processed_promptmd5(problem['statement'], template)
        rst = None
        for t in problem.get("processed",[]):
            if t["prompt_md5"][:8] == md5:
                rst = t["result"]
        if rst is not None:
            rsts.append(rst)
    
    rsts.sort(key=len)
    for idx, rst in enumerate(rsts):
        markdown += f'### 简要题意 {idx+1}\n\n{rst}\n\n'
    
    if markdown != '':
        markdown += '<br/>\n\n'
    
    markdown += f'### 原始题面\n\n```python\n{statement}\n```'
    return html, markdown
# 创建单一语言界面
def create_interface():
    with gr.Blocks(
        title="CTFcrypto原题机", css="""
        .mymarkdown {font-size: 15px !important}
        footer{display:none !important}
        .centermarkdown{text-align:center !important}
        .pagedisp{text-align:center !important; font-size: 20px !important}
        .realfooter{color: #888 !important; font-size: 14px !important; text-align: center !important;}
        .realfooter a{color: #888 !important;}
        .smallbutton {min-width: 30px !important;}
        """
    ) as demo:
        gr.Markdown("""
# 原题机
原题在哪里啊，原题在这里~"""
        )
        with gr.Tabs() as tabs:
            with gr.TabItem("搜索", id=0):
                input_text = gr.TextArea(
                    label="题目描述",
                    info="在这里粘贴你要搜索的题目！",
                    value="m = pow(c,d,n)",
                )
                bundles = []
                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("输入的问题描述将被重写并计算与每个原问题的最大相似度。")
                    for template_id in range(1): # 减少模板数量为2个
                        with gr.Accordion("版本 "+str(template_id+1)):
                            with gr.Row():
                                with gr.Group():
                                    template = settings.get('TEMPLATES', settings['TEMPLATES'])[template_id]
                                    engines = ["保留原描述", "deepseek-R1-32B"]
                                    engine = gr.Radio(
                                        engines,
                                        label="使用的语言模型",
                                        value=engines[1],
                                        interactive=True,
                                    )
                                    prompt = gr.TextArea(
                                        label="提示词 ([[ORIGINAL]] 将被替换为问题描述)",
                                        value=template,
                                        interactive=True,
                                        visible=True,
                                    )
                                    prefix = gr.Textbox(
                                        label="回复前缀",
                                        value="Simplified statement:",
                                        interactive=True,
                                        visible=True,
                                    )
                                    # 控制可见性
                                    engine.change(lambda engine: (gr.update(visible=not any(s in engine.lower() for s in ['保留', '跳过'])),)*2, engine, [prompt, prefix])
                                output_text = gr.TextArea(
                                    label="重写结果",
                                    value="",
                                    interactive=False,
                                )
                        bundles.append((engine, prompt, prefix, output_text))
                search_result = gr.State(([],[]))
                submit_button = gr.Button("搜索！")
                status_text = gr.Markdown("", elem_classes="centermarkdown")
            with gr.TabItem("查看结果", id=1):
                cur_idx = gr.State(0)
                num_columns = gr.State(1)
                
                # # 获取OJ列表 ctf的比赛名称简直是地狱绘图。
                # if hasattr(db, 'sources') and db.sources:
                #     ojs = [f'{t} ({c})' for t,c in sorted(db.sources.items())]
                # else:
                #     ojs = []
                
                # oj_dropdown = gr.Dropdown(
                #     ojs, value=ojs, multiselect=True, label="展示的OJ",
                #     info="不在这个列表里的OJ的题目将被忽略。可以在这里删掉你不认识的OJ。",
                # )
                
                # 重置索引
                #oj_dropdown.change(lambda: 0, None, cur_idx)
                statement_min_len = gr.Slider(
                    minimum=1,
                    maximum=2000,
                    label="最小题面长度",
                    value=20,
                    info="去除数字和空白字符后题面长度小于该值的题目将被忽略。可以用来筛掉一些奇怪的题面。",
                )

                with gr.Row():
                    add_column = gr.Button("+", elem_classes='smallbutton')
                    prev_page = gr.Button("←", elem_classes='smallbutton')
                    home_page = gr.Button("H", elem_classes='smallbutton')
                    next_page = gr.Button("→", elem_classes='smallbutton')
                    remove_column = gr.Button("-", elem_classes='smallbutton')
                    
                    # 页面导航功能
                    prev_page.click(lambda cur_idx, num_columns: max(cur_idx - num_columns, 0), [cur_idx, num_columns], cur_idx, concurrency_limit=None)
                    next_page.click(lambda cur_idx, num_columns: cur_idx + num_columns, [cur_idx, num_columns], cur_idx, concurrency_limit=None)
                    home_page.click(lambda: 0, None, cur_idx, concurrency_limit=None)
                    
                    def adj_idx(idx, col):
                        return int(round(idx / col)) * col
                        
                    add_column.click(lambda cur_idx, num_columns: (adj_idx(cur_idx, num_columns + 1), num_columns + 1), [cur_idx, num_columns], [cur_idx, num_columns], concurrency_limit=None)
                    remove_column.click(lambda cur_idx, num_columns: (adj_idx(cur_idx, num_columns - 1), num_columns - 1) if num_columns >1 else (cur_idx, num_columns), [cur_idx, num_columns], [cur_idx, num_columns], concurrency_limit=None)

                @gr.render(inputs=[search_result,  cur_idx, num_columns, statement_min_len], concurrency_limit=None)
                def show_OJs(search_result, cur_idx, num_columns, statement_min_len):
                    
                    gr.Markdown(f'第 {round(cur_idx/num_columns)+1} 页 / 共 {'2000+'} 页 (每页显示 {num_columns} 个)',
                               elem_classes="pagedisp")
                    cnt = 0
                    with gr.Row():
                        for sim, idx in zip(search_result[0], search_result[1]):
                            if not hasattr(db, 'metadata') or len(db.metadata) <= idx:
                                continue
                            if len(db.metadata[idx]) < 3 or db.metadata[idx][2] < statement_min_len:  # 移除OJ过滤条件
                                continue
                            cnt += 1
                            if cur_idx+1 <= cnt:
                                if cnt > cur_idx+num_columns: break
                                with gr.Column(variant='compact'):
                                    html, md = format_problem(idx, sim)
                                    gr.HTML(html)
                                    gr.Markdown(
                                        latex_delimiters=[
                                            {"left": "$$", "right": "$$", "display": True},
                                            {"left": "$", "right": "$", "display": False},
                                            {"left": "\\(", "right": "\\)", "display": False},
                                            {"left": "\\[", "right": "\\]", "display": True},
                                        ],
                                        value=md,
                                        elem_classes="mymarkdown",
                                    )

        # 页脚
        gr.HTML(
            """<div class="realfooter">Adapted from <a href="https://github.com/fjzzq2002/is-my-problem-new">https://github.com/fjzzq2002/is-my-problem-new</a></div><br /><div class="realfooter">Built with 💦 by <a href="https://github.com/LiErJ3412">LiErJ</a></div> """
        )
        
        # 搜索功能
        async def async_querier_wrapper(*args):
            result = await querier(*args)
            return (gr.Tabs(selected=1),) + tuple(result)
        
        submit_button.click(
            fn=async_querier_wrapper,
            inputs=sum([list(t[:-1]) for t in bundles], [input_text]),
            outputs=[tabs, status_text, search_result] + [t[-1] for t in bundles],
            concurrency_limit=7,
        )
    return demo

# 创建FastAPI应用
app = FastAPI()
favicon_path = 'favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

# 挂载Gradio应用
app = gr.mount_gradio_app(app, create_interface(), path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 改为8000端口