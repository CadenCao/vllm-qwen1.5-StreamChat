# -*- coding:utf-8 -*-
# Author:Canden Cao 
# time：2024-04-15
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig
import uvicorn,uuid,json,argparse

app=FastAPI()
userId_runId=dict()

# 采用vllm框架载入模型
def model_load_vllm(model_path,
                    max_model_len,
                    tensor_parallel_size,
                    quantization,
                    gpu_memory_utilization,
                    dtype):
    # 相关参数载入
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, max_model_len=max_model_len)
    generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, max_model_len=max_model_len)
    tokenizer.eos_token_id = generation_config.eos_token_id
    stop_tokens = [tokenizer.convert_tokens_to_ids('<|im_start|>'),
                      tokenizer.convert_tokens_to_ids('<|im_end|>'),
                      tokenizer.eos_token_id]
    """
    相关资料参考:
    https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
    https://docs.vllm.ai/en/v0.3.0/dev/engine/async_llm_engine.html
    https://zhuanlan.zhihu.com/p/649974825（gpu_memory_utilization）
    https://zhuanlan.zhihu.com/p/685884971
    https://zhuanlan.zhihu.com/p/678611154
    https://docs.vllm.ai/en/stable/serving/distributed_serving.html（tensor_parallel_size）
    https://qwen.readthedocs.io/en/latest/deployment/vllm.html（max_model_len）
    https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    """
    # AsyncEngineArgs参数载入（资料参考：https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py）
    args =AsyncEngineArgs(model_path,
                          tokenizer=model_path,
                          worker_use_ray=False,
                          engine_use_ray=False,
                          trust_remote_code=True,
                          tensor_parallel_size=tensor_parallel_size,
                          quantization=quantization,
                          dtype=dtype,
                          max_model_len=max_model_len,
                          gpu_memory_utilization=gpu_memory_utilization,
                          max_num_seqs=20
                          )
    engine = AsyncLLMEngine.from_engine_args(args)
    return generation_config, tokenizer, stop_tokens, engine

#问题及历史问题生成tokens
class query2token:
    def __init__(self,query,tokenizer,max_model_len,history):
        self.query=query
        self.tokenizer=tokenizer
        self.max_model_len=max_model_len
        self.history=history

    def __call__(self):
        start, start_token = '<|im_start|>', [self.tokenizer.convert_tokens_to_ids('<|im_start|>')]
        end, end_token = '<|im_end|>', [self.tokenizer.convert_tokens_to_ids('<|im_end|>')]
        nc,nc_token ='\n',self.tokenizer.encode("\n")
        system='<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
        system_token=start_token+self.tokenizer.encode('system')+nc_token+self.tokenizer.encode('You are a helpful assistant.')+end_token
        '''
        多轮模板(资料参考:https://zhuanlan.zhihu.com/p/678611154)
        """<|im_start|>system\n{system}<|im_end|>\n \
        <|im_start|>user\n{query1}<|im_end|>\n<|im_start|>assistant\n{response1}<|im_end|>\n \
        <|im_start|>user\n{query2}<|im_end|>\n<|im_start|>assistant\n"""
        .format(
            system="You are a helpful assistant.",
            query1="用户的第一次输入",
            response1="智能助手的第一次回复",
            query2="用户的第二次输入"
        )
        '''
        prompt_list,prompt_token_list=[],[]
        prompt_list.append(system)
        prompt_token_list.append(system_token)
        user_token, assistant_token = self.tokenizer.encode('user'), self.tokenizer.encode('assistant')
        if self.history:
            for query,response in self.history[::-1]:
                query_token,response_token=self.tokenizer.encode(query),self.tokenizer.encode(response)
                str_=f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n'
                token_=start_token+user_token+nc_token+query_token+end_token\
                       +nc_token+start_token+assistant_token+nc_token+response_token+end_token+nc_token
                prompt_list.append(str_)
                prompt_token_list.append(token_)
        str_,query_token = f'<|im_start|>user\n{self.query}<|im_end|>\n<|im_start|>assistant\n',self.tokenizer.encode(self.query)
        token_ = start_token + user_token + nc_token + query_token + end_token \
                 + nc_token + start_token + assistant_token + nc_token
        prompt_list.append(str_)
        prompt_token_list.append(token_)

        prompt,prompt_token,token_count='',[],0
        for i in range(len(prompt_list)):
            prompt_,prompt_token_=prompt_list[i],prompt_token_list[i]
            if i>0 and token_count+len(prompt_token_)>self.max_model_len:
                break
            prompt+=prompt_
            prompt_token+=prompt_token_
            token_count += len(prompt_token_)
        return prompt,prompt_token


@app.post("/stream_chat")
async def chat(request: Request):
    global generation_config, tokenizer, stop_tokens, engine
    request = await request.json()
    query = request.get('query', None)
    chat_aim=request.get('chat_aim',False)
    userId=request.get('userId','admin')
    history = request.get('history', [])
    gradio=request.get('gradio', True)

    if chat_aim:
        query_to_token=query2token(query,tokenizer,args.max_model_len,history)
        prompt, prompt_tokens=query_to_token()

        # 每一个用户的协程唯一标识
        request_id = str(uuid.uuid4().hex)
        userId_runId[userId] = request_id
        # vLLM请求配置 （资料参考:https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py）
        sampling_params = SamplingParams(top_p=generation_config.top_p,
                                         top_k=generation_config.top_k,
                                         early_stopping=False,
                                         temperature=generation_config.temperature,
                                         repetition_penalty=generation_config.repetition_penalty,
                                         max_tokens=generation_config.max_new_tokens,
                                         stop_token_ids=stop_tokens)
        # 参数资料参考:https://docs.vllm.ai/en/v0.3.0/dev/engine/async_llm_engine.html
        generater = engine.generate(prompt=None,
                                    sampling_params=sampling_params,
                                    prompt_token_ids=prompt_tokens,
                                    request_id=request_id)

    # 采用gradio组件方式展示
    if gradio:
        if chat_aim:
            async def streaming_resp():
                async for item in generater:
                    tokens=item.outputs[0].token_ids
                    # 去除终止符号
                    if tokens[-1] in stop_tokens:
                        tokens.pop()
                    text = tokenizer.decode(tokens)
                    yield (json.dumps({'text': text}) + '\0').encode('utf-8')
                await engine.abort(request_id)
            return StreamingResponse(streaming_resp())
        else:
            await engine.abort(userId_runId[userId])
    # 采用命令行方式展示
    else:
        async def streaming_resp():
            async for item in generater:
                tokens=item.outputs[0].token_ids
                if tokens[-1] in stop_tokens:
                    tokens.pop()
                # 返回截止目前的tokens输出
                text = tokenizer.decode(tokens)
                yield (json.dumps({'text': text}) + '\0').encode('utf-8')
            await engine.abort(userId_runId[userId])
        return StreamingResponse(streaming_resp())


def argument_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',default='../FAQS/Qwen1.5-14B-Chat-GPTQ-Int4',type=str,help='模型路径')
    parser.add_argument('--tensor_parallel_size',default=1,type=int,help='模型运行需要的GPU数量')
    parser.add_argument('--quantization', default='gptq', type=str, help='量化方式')
    parser.add_argument('--gpu_memory_utilization',default=0.9,type=float,help='GPU剩余利用率')
    parser.add_argument('--dtype', default='auto', type=str,help='加载数据类型')
    parser.add_argument('--max_model_len', default=2000, type=int, help='最大模型的')
    parser.add_argument('--host', default='0.0.0.0', type=str, help='主机地址')
    parser.add_argument('--port', default=5499, type=int, help='端口号')
    args=parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    generation_config, tokenizer, stop_tokens, engine = model_load_vllm(args.model_path,
                                                                        args.max_model_len,
                                                                        args.tensor_parallel_size,
                                                                        args.quantization,
                                                                        args.gpu_memory_utilization,
                                                                        args.dtype)
    uvicorn.run(app,host=args.host,port=args.port,log_level="debug")