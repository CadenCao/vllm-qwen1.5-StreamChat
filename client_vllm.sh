python client_vllm.py \
          --mode='gradio' \
          --sever_url='http://0.0.0.0:5499/stream_chat' \
          --history_max=5 \
          --model_name='Qwen1.5-14B-Chat-GPTQ-Int4' \
          --concurrency_count=5 \
          --host='0.0.0.0' \
          --port=5500