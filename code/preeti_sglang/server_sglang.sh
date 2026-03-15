python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --mem-fraction-static 0.8\
      --disable-cuda-graph --host 0.0.0.0 --attention-backend flashinfer\
      --max-running-requests 50 \
      # --schedule-policy lpm \
      # --enable-metrics \

# nv-nsight-cu-cli --target-processes all --launch-skip 155 --launch-count 200 \
#       --metrics lts__t_request_hit_rate.pct,lts__t_requests.sum,idc__request_hit_rate.pct,idc__requests.sum --csv \
#       python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --mem-fraction-static 0.8 \
#       --disable-cuda-graph --host 0.0.0.0 --attention-backend flashinfer

# lts__t_request_hit_rate.pct,lts__t_requests.sum,idc__request_hit_rate.pct,idc__requests.sum,l1tex__t_requests.sum

# nv-nsight-cu-cli --target-processes all --section MemoryWorkloadAnalysis --launch-skip 200 --launch-count 50 --csv \
#       python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --mem-fraction-static 0.8 \
#       --disable-cuda-graph --host 0.0.0.0 --attention-backend flashinfer --enable-metrics

#  --enable-metrics
# --disable-cuda-graph
# --log-requests --log-requests-level 1
# --enable-mixed-chunk --chunked-prefill-size 256 
# --enable-request-time-stats-logging #Req Time Stats(rid=23b2f003434b476aa6d3bf6706e710c0, input len=38, output len=484, type=unified): queue_duration=0.00ms, forward_duration=1759654443392.38ms, start_time=0.0
# --enable-two-batch-overlap 
# --decode-log-interval 10 
# --disable-overlap-schedule # to disable cpu-gpu overlapping --> scheduler-->event_loop_normal else event_loop_overlap
# --enable-lmcache
# --schedule-policy {lpm,random,fcfs,dfs-weight,lof}

# qwen/qwen2.5-0.5b-instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# meta-llama/Llama-3.2-1B-Instruct

# nsys profile -t cuda -o nsys_res -f true --trace-fork-before-exec=true --cuda-graph-trace=node --delay 20 --duration 100 python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.7 --disable-cuda-graph --host 0.0.0.0

# nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --disable-radix-cache

# nsys profile --trace=cuda,nvtx,osrt -o sglang_profile python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --disable-radix-cache

# ncu --target-processes all \
#     --devices 0,1 \
#     --set full \
#     -k regex:.*kernel.* \
#     --kernel-name-base demangled \
#     --launch-skip 50 \
#     --launch-count 5 \
#     python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.7 --disable-cuda-graph --host 0.0.0.0 --attention-backend flashinfer

# ncu --call-stack python python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.7 --disable-cuda-graph --host 0.0.0.0 --attention-backend flashinfer

# ncu --target-processes all \
#     --metrics l2_tex_hit_rate,l2_tex_read_hit_rate,l2_tex_write_hit_rate \
#     --set full \
#     --launch-skip 5 --launch-count 10 \
#     -o sglang_cache \
#     python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --disable-radix-cache

# ncu --target-processes all \
#     --metrics l2_tex_hit_rate,l2_tex_read_hit_rate,l2_tex_write_hit_rate \
#     --set full \
#     --launch-skip 5 --launch-count 10 \
#     -o sglang_cache \
#     bash -c "python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --disable-radix-cache & \
#              sleep 30 && \
#              python3 cl.py && \
#              sleep 10"

# python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --num-prompts 50 --model qwen/qwen2.5-0.5b-instruct \
#  --dataset-name sharegpt --sharegpt-output-len 100 --request-rate 10 --profile --flush-cache

# python -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.8 --quantization fp8 \
#  --disable-cuda-graph --enable-metrics --host 0.0.0.0

#  nsys profile -t cuda --cuda-graph-trace=node --cuda-memory-usage true -o req1846_rr426_profiler.out --delay 20 --duration 100 python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.8 --disable-cuda-graph --enable-metrics --host 0.0.0.0

# nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column --output req1846_rr284_profiler.csv req1846_rr284_profiler.out.nsys-rep

# nsys profile --trace=cuda,osrt,nvtx --cuda-memory-usage true --stats true --force-overwrite true --output mm3_nsys_2048 ./mm3 2048
# nsys stats --report osrtsum --report gputrace --report gpukernsum --report gpumemsizesum --report gpumemtimesum --report cudaapisum --format csv --output mm3_nsys_2048 mm3_nsys_2048.nsys-rep 

# nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,memory.reserved,pcie.link.gen.gpucurrent,pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max --format=csv -lms 100 -f gpu_smi_log_4096.csv

# FLOPs=(FADD+FMUL+DADD+DMUL+HADD+HMUL)×1+(FFMA+DFMA+HFMA)×2+(Tensor instructions)×FLOPs per tensor instruction

# ncu --metrics sm__sass_thread_inst_executed_op_hfma_pred_on, \
#     sm__sass_thread_inst_executed_op_hadd_pred_on, \
#     sm__sass_thread_inst_executed_op_hmul_pred_on, \
#     sm__sass_thread_inst_executed_op_ffma_pred_on, \
#     sm__sass_thread_inst_executed_op_fadd_pred_on, \
#     sm__sass_thread_inst_executed_op_fmul_pred_on, \
#     sm__sass_thread_inst_executed_op_dadd_pred_on, \
#     sm__sass_thread_inst_executed_op_dfma_pred_on, \
#     sm__sass_thread_inst_executed_op_dmul_pred_on, \
#     sm__inst_executed_pipe_tensor_op_hmma, \
#     sm__inst_executed_pipe_tensor_op_imma, \
#     sm__sass_thread_inst_executed_op_fp16_pred_on, \
#     sm__sass_thread_inst_executed_op_fp32_pred_on, \
#     sm__sass_thread_inst_executed_op_fp64_pred_on, \
#     sm__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on, \
#     sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on, \
#     sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on -c 1 ./mm3 128

# ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy -c 1 ./mm3 2048