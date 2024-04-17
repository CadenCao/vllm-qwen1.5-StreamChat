# vllm框架+Qwen1.5系列模型+流式输出+多并发
***
### 一、利用vllm框架加载Qwen1.5并进行了流式输出（支持多并发） 
* Gradio效果展示

![采用gradio组件展示](./pictures/qwen_gradio.gif 'qwen_gradio')

* 命令行效果展示

![采用命令行展示](./pictures/qwen_web.gif 'qwen_web')

***
### 二、模型加载  

需要分别启动服务端和客户端
1. 服务端启动  
服务端共有两个文件，分别为`server_vllm.py`和`servervllm.sh`，代表直接运行和shell启动。
   * 直接运行  
   直接运行`server_vllm.py`文件，配置参数需要在`server_vllm.py`文件内部修改
   * shell启动  
   直接运行`sh server_vllm.sh`命令，配置参数需要在`server_vllm.sh`文件内部修改
   
     
2. 客服端启动
客户端共有两个文件，分别为`client_vllm.py`和`client_vllm.sh`，代表直接运行和shell启动。
   * 直接运行  
    直接运行`client_vllm.py`文件，配置参数需要在`client_vllm.py`文件内部修改
    * shell启动  
    直接运行`sh client_vllm.sh`命令，配置参数需要在`client_vllm.sh`文件内部修改
   
***
### 三、参数说明
1. 服务端参数说明
```shell
# CUDA_VISIBLE_DEVICES需要的GPU代号
CUDA_VISIBLE_DEVICES=0 python server_vllm.py \
          # 模型路径
          --model_path='./Qwen1.5-14B-Chat-GPTQ-Int4' \
          
          # 使用GPU的数量
          --tensor_parallel_size=1 
          
          # 用于量化权重的方法。如果为None，模型首先检查'quantiation_config'属性，
          # 如果是None，则认为模型权重不是量化并使用'dtype'来确定数据权重的类型。
          --quantization='gptq' \
          
          # GPU剩余内存用于运行大模型的比例，取值在0-1，默认是0.9
          --gpu_memory_utilization=0.9 \
          
          # 模型权重数据类型，可以取['auto', 'half', 'float16', 'bfloat16', 'float', 'float32']
          --dtype='auto' \
          
          # 输入的上下文长度，系统默认是32768        
          --max_model_len=2000 \
          
          # 主机地址
          --host='0.0.0.0' \
          
          # 主机端口
          --port=5499
```
* 当出现了模型内存不足时，可以尝试增大`gpu_memory_utilization`和降低`max_model_len`
* 使用`Qwen1.5`的`7B`和`14B`模型时，必须选择`24G`显存以上的显卡，`Qwen1.5-14B-Chat-GPTQ-Int4`单张16G的T4可以满足
  
2. 服务端参数说明
```shell
python server_vllm.py \
          # 客户端展示方法，mode值为''代表为采用命令行方式运行，mode值为'gradio`表示以gradio方式展示
          --mode='' \
          
          # 大模型运行的服务端地址
          --sever_url='http://0.0.0.0:5499/stream_chat' \
          
          # 多轮对话，当history_max=0时表示仅为单轮对话，为n表示最近的n轮对话。注意，该值需要和
          # max_model_len参数配合使用。
          --history_max=5 \
          
          # 使用gradio展示时，首页展示模型的名称，可以自定义。
          --model_name='Qwen1.5-14B-Chat-GPTQ-Int4' \
          
          # 当使用gradio时采用的线程数
          --concurrency_count=5 \
          
          # gradio服务的服务IP地址
          --host='0.0.0.0' \
          
          # gradio服务的端口
          --port=5499
```
### 四、参考资料 

https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
https://docs.vllm.ai/en/v0.3.0/dev/engine/async_llm_engine.html
https://zhuanlan.zhihu.com/p/649974825
https://zhuanlan.zhihu.com/p/678611154
https://docs.vllm.ai/en/stable/serving/distributed_serving.html
https://qwen.readthedocs.io/en/latest/deployment/vllm.html
https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
