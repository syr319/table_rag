import os
import json
import pandas as pd
import re
import fire
from evaluate import evaluate

def evaluate_from_logs(
    log_dir,
    task=None
):
    # 1. 自动识别任务类型
    if not task:
        config_path = os.path.join(os.path.dirname(log_dir), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                task = config.get('task')
                print(f"从配置文件识别任务类型：{task}")
        else:
            raise ValueError("未找到配置文件，请通过 --task 指定任务类型（如 'hitab'）")

    # 2. 加载 sc_id=4 的日志文件，并补充 id 字段
    results = []
    # 匹配文件名格式：xxx-数字-4.json（提取数据ID）
    pattern = re.compile(r'^(.*?)-\d+\.json$')  # 例如从 hitab_test-48-4.json 提取 hitab_test-48
    # log_files = [f for f in os.listdir(log_dir) if f.endswith('-4.json') and f.endswith('.json')]
    log_files = [f for f in os.listdir(log_dir) if  f.endswith('.json')]
    if not log_files:
        raise FileNotFoundError(f"日志目录 {log_dir} 中未找到 sc_id=4 的 .json 日志文件")
    
    for file in log_files:
        file_path = os.path.join(log_dir, file)
        with open(file_path, 'r') as f:
            log_data = json.load(f)
        
        # 从文件名提取数据ID（若日志中无 'id' 字段）
        match = pattern.match(file)
        if match and 'id' not in log_data:
            data_id = match.group(1)  # 提取 hitab_test-48 作为 id
            log_data['id'] = data_id
            print(f"从文件名补充 id: {data_id}（文件：{file}）")
        
        # 确保包含评估所需字段
        results.append({
            'id': log_data.get('id', 'unknown'),  # 优先用日志中的id，否则设为unknown
            'answer': log_data['answer'],
            'label': log_data['label']
        })
    
    print(f"成功加载 {len(results)} 条 sc_id=4 的日志数据")

    # 3. 调用评估函数
    acc = evaluate(task, results)
    print(f"准确率 (Accuracy): {acc}")

    # 4. 生成统计信息
    stats_df = pd.DataFrame(results)
    print("\n预测结果统计：")
    print(stats_df['answer'].describe())

    return acc

if __name__ == '__main__':
    fire.Fire(evaluate_from_logs)
    