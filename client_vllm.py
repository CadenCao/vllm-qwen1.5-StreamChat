# -*- coding:utf-8 -*-
# Author:Canden Cao 
# time：2024-04-16
import gradio as gr
import requests
import json,argparse


def client(**kwagrs):
    if kwagrs['mode']=='gradio':
        def submitBtn_fn(user_input, chatbot, history,history_max):
            chatbot.append((user_input, ""))
            request_get = requests.post(kwagrs['sever_url'],
                                     json={'query':user_input,
                                            'chat_aim':True,
                                            'userId':'admin',
                                            'history':history[-history_max:] if history_max>0 else [],
                                            'gradio':False,
                                            }, stream=True)
            for chunk in request_get.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode('utf-8'))
                    respone = data["text"].rstrip('\r\n')
                    chatbot[-1]=(user_input,respone)
                    history[-1]=(user_input, respone) if history else history.append((user_input, respone))
                    yield chatbot, history


        def reset_user_input():
            return gr.Textbox.update(show_label=False, value=None, placeholder="请输入您的问题",lines=5,container=False)

        def stop_fn(user_name):
            requests.post(kwagrs['sever_url'],
                          json={'query': None,
                                'chat_aim': False,
                                'userId': user_name,
                                'history': [],
                                'gradio': True,
                                }, stream=True)

        def emptyBtn_fn():
            return [], []

        with gr.Blocks() as demo:
            gr.HTML(f"""<h1 align="center">{kwagrs['model_name']}</h1>""")
            with gr.Row():
                with gr.Column(scale=1):
                    user_name = gr.Textbox(label='用户名', placeholder="请输入您的用户名", lines=1,container=False,value='admin')
                with gr.Column(scale=4):
                    pass
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                with gr.Column(scale=5):
                    user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题", lines=5,container=False)
                    submitBtn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        stop_gen = gr.Button("停止生成")
                        emptyBtn = gr.Button("清除历史消息")
                    history_max = gr.Slider(0, 5, value=5, step=1, label="多对话轮数", interactive=True,visible=False)
                    markdown=gr.Markdown("""
                    **注意**：  
                    当需要多人同时访问时，请输入您的用户名，不然"停止生成"可能会终止别的用户输出！ 
                    """,visible=True)

            history = gr.State([])
            submitBtn.click(submitBtn_fn, [user_input, chatbot, history,history_max],[chatbot, history],show_progress=True)
            submitBtn.click(reset_user_input, [],[user_input])
            stop_gen.click(stop_fn, inputs=[user_name])
            emptyBtn.click(emptyBtn_fn, outputs=[chatbot,history])

        demo.queue(concurrency_count=kwagrs['concurrency_count']).launch(server_port=kwagrs['port']
                                                                         ,server_name=kwagrs['host']
                                                                         ,share=False)

    else:
        history = []
        while True:
            query = input('请输入您的问题：')
            request_get = requests.post(kwagrs['sever_url'],
                                     json={'query':query,
                                            'chat_aim':True,
                                            'userId':'admin',
                                            'history':history,
                                            'gradio':False,
                                            }, stream=True)

            respone=''
            for chunk in request_get.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode('utf-8'))
                    respone_temp = data["text"].rstrip('\r\n')
                    for i in respone_temp[len(respone):]:
                        print(i,end='')
                    respone=respone_temp
            print('\n')
            history.append((query, respone))
            history = history[-kwarg['history_max']:] if kwarg['history_max']>0 else []


def argument_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',default='',type=str,help='展示方式，值为gradio或空')
    parser.add_argument('--sever_url',default='http://0.0.0.0:5499/stream_chat',type=str,help='大模型运行地址')
    parser.add_argument('--history_max',default=5,type=int,help='多轮对话数')
    parser.add_argument('--model_name', default='Qwen1.5-14B-Chat-GPTQ-Int4', type=str,help='模型名称')
    parser.add_argument('--concurrency_count', default=5, type=int, help='线程数')
    parser.add_argument('--host', default='0.0.0.0', type=str, help='主机地址')
    parser.add_argument('--port', default=5500, type=int, help='端口号')
    args=parser.parse_args()
    return args


if __name__=="__main__":
    args = argument_parser()
    kwarg={
        'mode':args.mode,
        'sever_url':args.sever_url,
        'history_max':args.history_max,
        'model_name':args.model_name,
        'concurrency_count':args.concurrency_count,
        'host':args.host,
        'port':args.port
    }
    client(**kwarg)