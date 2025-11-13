# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import warnings
import logging  # 导入日志模块
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import pandas as pd
from tqdm import tqdm

from agent import TableAgent, TableRAGAgent
from evaluate import evaluate
from utils.load_data import load_dataset


# -------------------------- 日志配置（新增） --------------------------
def init_logger(log_dir):
    """初始化日志：同时输出到文件和控制台"""
    # 日志文件路径（按时间命名，避免覆盖）
    log_file = os.path.join(log_dir, "run.log")
    
    # 清除已有日志处理器（避免重复输出）
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志格式（包含：时间戳、日志级别、模块名、日志内容）
    log_format = "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    logging.basicConfig(
        level=logging.INFO,  # 日志级别：INFO及以上（包含模型输出、进度、错误）
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # 写入日志文件（UTF-8兼容中文）
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logging.info(f"日志已初始化，文件保存路径：{log_file}")
    return logging


def solve(args):
    agent_args, data, sc_id = args
    logger = logging.getLogger()  # 获取全局日志器
    try:
        if 'TableRAG' in agent_args['agent_type']:
            agent = TableRAGAgent(**agent_args)
            logger.info(f"初始化 TableRAGAgent，sc_id={sc_id}")
        elif agent_args['agent_type'] in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
            agent = TableAgent(**agent_args)
            logger.info(f"初始化 {agent_args['agent_type']} Agent，sc_id={sc_id}")
        else:
            raise NotImplementedError(f"Agent type {agent_args['agent_type']} not supported.")
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # 记录当前处理的数据基本信息（如问题、表格ID）
            data_id = data.get('id', f"unknown_{sc_id}")
            logger.info(f"开始处理数据 [ID: {data_id}]，sc_id={sc_id}")
            result = agent.run(data, sc_id=sc_id)
        
        # 记录单条数据处理结果（包含推理步骤、答案等）
        logger.info(f"数据 [ID: {data_id}] 处理完成，结果摘要：n_iter={result.get('n_iter', -1)}, 预测答案={result.get('pred', 'None')}")
        return result
    
    except Exception as e:
        # 捕获异常并记录详细错误信息
        logger.error(f"处理数据 sc_id={sc_id} 时出错：{str(e)}", exc_info=True)
        raise  # 重新抛出异常，不影响整体任务中断


def main(
    dataset_path = 'data/tabfact/test_sub_nosynth.jsonl',
    model_name = 'gpt-3.5-turbo-0125',
    agent_type = 'PyReAct',
    retrieve_mode = 'embed',
    embed_model_name = 'text-embedding-3-large',
    log_dir = 'output/test',
    db_dir = 'db/',
    top_k = 5,
    sr = 0, # self-refine, deprecated
    sc = 1, # self-consistency
    max_encode_cell = 10000,
    stop_at = -1,
    resume_from = 0,
    load_exist = False,
    n_worker = 1,
    verbose = False,
):
    # 1. 初始化日志（新增）
    os.makedirs(log_dir, exist_ok=True)  # 确保日志根目录存在
    logger = init_logger(log_dir)  # 初始化日志器

    # 2. 打印并记录运行配置
    logger.info("="*50 + " 任务启动 " + "="*50)
    logger.info(f"运行参数：dataset_path={dataset_path}, model_name={model_name}, agent_type={agent_type}")

    os.makedirs(os.path.join(log_dir, 'log'), exist_ok=True)

    # 识别任务类型
    # task_names = ['tabfact', 'wtq', 'arcade', 'bird', 'hitab', 'tablebench']  
    # task = [task_name for task_name in task_names if task_name in dataset_path]  
    # # task = task_candidates[0] if task_candidates else 'hitab' 
    task = [task_name for task_name in ['tabfact', 'wtq', 'arcade', 'bird','hitab','tablebench'] if task_name in dataset_path][0]
    logger.info(f"自动识别任务类型：{task}")

    db_dir = os.path.join(db_dir, task + '_' + Path(dataset_path).stem)
    config_path = os.path.join(log_dir, 'config.json')
    # with open(config_path, 'w') as fp:
    #     config = {key: value for key, value in locals().items() if key != 'fp'}
    #     json.dump(config, fp, indent=4)

    # 修改后代码：明确指定需要保存的配置参数
    with open(config_path, 'w') as fp:
    # 只保留任务相关的关键参数，排除 logger 等模块对象
        config = {
        "dataset_path": dataset_path,
        "model_name": model_name,
        "agent_type": agent_type,
        "retrieve_mode": retrieve_mode,
        "embed_model_name": embed_model_name,
        "log_dir": log_dir,
        "db_dir": db_dir,
        "top_k": top_k,
        "sr": sr,
        "sc": sc,
        "max_encode_cell": max_encode_cell,
        "stop_at": stop_at,
        "resume_from": resume_from,
        "load_exist": load_exist,
        "n_worker": n_worker,
        "verbose": verbose,
        "task": task  # 加入自动识别的 task 变量
    }
        json.dump(config, fp, indent=4)

    logger.info(f"配置文件已保存：{config_path}")

    # 3. 加载数据集
    logger.info(f"开始加载数据集：{dataset_path}")
    dataset = load_dataset(task, dataset_path, stop_at)
    if stop_at < 0:
        stop_at = len(dataset)
    logger.info(f"数据集加载完成，共 {len(dataset)} 条数据，实际处理范围：[{resume_from}, {stop_at})")

    # 4. 准备Agent参数
    agent_args = {
        'model_name': model_name,
        'retrieve_mode': retrieve_mode,
        'embed_model_name': embed_model_name,
        'task': task,
        'agent_type': agent_type,
        'top_k': top_k,
        'sr': sr,
        'max_encode_cell': max_encode_cell,
        'log_dir': log_dir,
        'db_dir': db_dir,
        'load_exist': load_exist,
        'verbose': verbose
    }
    logger.info(f"Agent参数准备完成：{json.dumps(agent_args, indent=2)}")

    # 5. 执行任务（单进程/多进程）
    results = []
    total_tasks = (stop_at - resume_from) * sc
    logger.info(f"开始执行任务，总任务数：{total_tasks}，进程数：{n_worker}")

    if n_worker == 1:
        for data in tqdm(dataset[resume_from:stop_at], desc="总进度"):
            for sc_id in tqdm(range(sc), position=1, leave=False, desc="Self-Consistency"):
                result = solve((agent_args, data, sc_id))
                results.append(result)
    else:
        with tqdm(total=total_tasks, desc="总进度") as pbar:
            with ProcessPoolExecutor(max_workers=n_worker) as executor:
                futures = [
                    executor.submit(solve, (agent_args, data, sc_id)) 
                    for data in dataset[resume_from:stop_at] 
                    for sc_id in range(sc)
                ]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())

    # 6. 评估并记录结果
    logger.info(f"所有任务执行完成，开始评估准确率")
    acc = evaluate(task, results)
    logger.info(f"准确率：{acc}")  # 日志记录准确率
    print(f'Accuracy: {acc}')  # 保持控制台输出习惯

    stats_keys = ['n_iter', 'init_prompt_token_count', 'total_token_count']
    stats_df = pd.DataFrame.from_records(results)[stats_keys]
    stats_desc = stats_df.describe().to_string()
    logger.info(f"统计信息：\n{stats_desc}")  # 日志记录统计信息
    print(stats_desc)  # 保持控制台输出习惯

    # 7. 保存结果文件
    result_dict = stats_df.mean().to_dict()
    result_dict['accuracy'] = acc
    for key in ['model_name', 'retrieve_mode', 'embed_model_name', 'task', 'agent_type', 'top_k', 'max_encode_cell', 'sr']:
        result_dict[key] = agent_args[key]
    result_dict['sc'] = sc
    result_dict['data'] = Path(dataset_path).stem
    result_path = os.path.join(log_dir, 'result.json')
    with open(result_path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)
    logger.info(f"结果文件已保存：{result_path}")
    logger.info("="*50 + " 任务结束 " + "="*50)


if __name__ == '__main__':
    fire.Fire(main)