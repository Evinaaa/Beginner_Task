import subprocess
import csv
import time
from itertools import product

# 实验配置
datasets = ['PROTEINS', 'MUTAG', 'ENZYMES']
gnn_types = ['gcn', 'gat', 'sage', 'gin']
pooling_types = ['avg', 'max', 'min']

# 结果文件
results_file = 'experiment_results.csv'
with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'GNN Type', 'Pooling Type', 'Test Accuracy', 'Time (s)'])

# 运行所有组合
for dataset, gnn, pooling in product(datasets, gnn_types, pooling_types):
    print(f'\n{"="*50}')
    print(f'开始实验: {dataset} | {gnn} | {pooling}')
    print(f'{"="*50}')
    
    start_time = time.time()
    
    # 运行训练脚本
    command = f"python train.py --dataset {dataset} --gnn {gnn} --pooling {pooling}"
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # 获取输出
    output = process.stdout
    elapsed = time.time() - start_time
    
    # 解析准确率 (从最后几行获取)
    test_acc = None
    for line in output.splitlines()[-5:]:
        if '测试准确率:' in line:
            test_acc = float(line.split(': ')[1].strip())
            break
    
    # 保存结果
    if test_acc is not None:
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dataset, gnn, pooling, test_acc, round(elapsed, 2)])
        
        print(f'完成: {dataset} | {gnn} | {pooling} | 准确率: {test_acc:.4f} | 时间: {elapsed:.2f}s')
    else:
        print(f'错误: 无法获取 {dataset} | {gnn} | {pooling} 的准确率')