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

import time
from typing import Optional

import tiktoken
import vertexai
from openai import OpenAI
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, HarmCategory, HarmBlockThreshold


PROJECT_ID = "YOUR_GCP_PROJECT_ID"
LOCATION = "YOUR_GCP_LOCATION"

# -------------------------- 新增：本地 vLLM 模型配置（核心修改） --------------------------
# 配置本地模型的关键信息：模型别名、实际tokenizer路径、上下文长度、vLLM服务端口
LOCAL_VLLM_CONFIG = {
    # Qwen2.5-3B-Instruct 配置
    "qwen_3b": {
        "tokenizer_name": "Qwen/Qwen2.5-3B-Instruct",  # 实际tokenizer路径
        "context_limit": 16384,  # Qwen2.5-3B默认上下文长度
        "vllm_port": 8889  # Qwen模型的vLLM服务端口（根据实际部署修改）
    },
    # TableLLM-7b 配置
    "tablellm-7b": {
        "tokenizer_name": "RUCKBReasoning/TableLLM-7b",  # 实际tokenizer路径
        "context_limit": 14784,  # TableLLM-7b常见上下文长度（需根据模型实际参数调整）
        "vllm_port": 8889  # TableLLM的vLLM服务端口（避免与Qwen冲突）
    }
    # 可在此处添加更多本地vLLM模型配置
}


class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.provider = self.get_provider(model_name)  # 'openai'/'google'/'vllm'
        self.vllm_config = self.get_vllm_config(model_name)  # 新增：vLLM模型专属配置
        self.context_limit = self.get_context_limit(model_name)
       

        # OpenAI models
        if self.provider == 'openai':
            self.client = OpenAI()
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        # Gemini models
        elif self.provider == 'google':
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self.client = GenerativeModel(model_name)
        # vLLM models（适配本地TableLLM和Qwen3）
        elif self.provider == 'vllm':
            # 根据模型配置获取vLLM服务端口
            vllm_port = self.vllm_config["vllm_port"]
            self.client = OpenAI(
                base_url=f"http://localhost:{vllm_port}/v1",  # 动态适配端口
                api_key="token-abc123",  # vLLM默认api_key，无需修改
            )
            # 加载对应模型的tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.vllm_config["tokenizer_name"],
                trust_remote_code=True  # 关键：TableLLM可能需要加载自定义代码
            )
            # # 补充：Qwen/TableLLM的tokenizer可能需要特殊配置
            # if "qwen" in self.vllm_config["tokenizer_name"].lower():
            #     self.tokenizer.pad_token = self.tokenizer.eos_token  # Qwen默认无pad_token
            # if "tablellm" in self.vllm_config["tokenizer_name"].lower():
            #     self.tokenizer.padding_side = "left"  # 按需调整TableLLM的padding方向


    def get_provider(self, model_name):
        """优化：优先识别本地vLLM模型别名"""
        if model_name in LOCAL_VLLM_CONFIG.keys():
            return 'vllm'
        elif 'gpt' in model_name:
            return 'openai'
        elif 'gemini' in model_name:
            return 'google'
        else:
            # 若输入完整tokenizer路径（如"Qwen/Qwen2.5-3B-Instruct"），也识别为vLLM
            return 'vllm'

    def get_vllm_config(self, model_name):
        """获取本地vLLM模型的配置（tokenizer/端口/上下文长度）"""
        # 若输入的是模型别名（如"qwen_3b"），直接返回配置
        if model_name in LOCAL_VLLM_CONFIG.keys():
            return LOCAL_VLLM_CONFIG[model_name]
        # 若输入的是完整tokenizer路径（如"Qwen/Qwen2.5-3B-Instruct"），匹配配置
        for alias, config in LOCAL_VLLM_CONFIG.items():
            if config["tokenizer_name"] == model_name:
                return config
        # 未匹配到预设配置时，返回默认值（可按需扩展）
        raise ValueError(f"本地vLLM模型 {model_name} 未在 LOCAL_VLLM_CONFIG 中配置")

    def get_context_limit(self, model_name):
        """优化：优先读取本地vLLM模型的上下文长度"""
        # 本地vLLM模型从配置中获取
        if self.provider == 'vllm':
            return self.vllm_config["context_limit"]
        # 原有模型的上下文长度配置
        elif model_name == 'gpt-4-0125-preview' or model_name == 'gpt-4-turbo-2024-04-09' or model_name == 'gpt-4o-mini-2024-07-18':
            return 128000
        elif model_name == 'gpt-3.5-turbo-0125':
            return 16385
        elif model_name == 'gemini-pro' or model_name == 'gemini-ultra':
            return 32000
        elif model_name == 'gemini-1.5-pro-preview-0409' or model_name == 'gemini-1.5-flash':
            return 128000
        elif 'Mistral-Nemo' in model_name:
            return 128000
        else:
            raise ValueError(f'Unsupported model: {model_name}')

    def query(self, prompt, **kwargs):
        if not prompt:
            return 'Contents must not be empty.'
        if self.provider == 'openai':
            return self.query_openai(prompt,** kwargs)
        elif self.provider == "google":
            return self.query_gemini(prompt, **kwargs)
        elif self.provider == "vllm":
            # vLLM复用OpenAI接口格式（兼容chat completions）
            return self.query_openai(prompt, **kwargs)
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_gemini_with_retry(self, prompt, generation_config):
        safety_config = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = self.client.generate_content(prompt, generation_config=generation_config, safety_settings=safety_config)
        try:
            response_text = response.text
        except Exception as e:
            response_text = str(e)
        return response_text

    def query_gemini(self, prompt, rate_limit_per_minute = None, **kwargs):
        generation_config = GenerationConfig(
            stop_sequences=kwargs.get('stop', []),
            temperature=kwargs.get('temperature'),
            top_p=kwargs.get('top_p'),
        )
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
        return self.query_gemini_with_retry(prompt, generation_config=generation_config)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_openai_with_retry(self, messages, **kwargs):
        # 补充：vLLM可能需要显式指定max_tokens（避免默认值过小）
        if self.provider == 'vllm' and 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 2048  # 按需调整
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,** kwargs
        )

    def query_openai(self,
                     prompt,
                     system = None,
                     rate_limit_per_minute = None, **kwargs):
        # Set default system message
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        response = self.query_openai_with_retry(messages, **kwargs)
        # Sleep to avoid rate limit if rate limit is set
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_gemini_token_count(self, prompt):
        return self.client.count_tokens(prompt).total_tokens

    def get_token_count(self, prompt):
        if not prompt:
            return 0
        if self.provider == 'openai':
            return len(self.tokenizer.encode(prompt))
        elif self.provider == "google":
            return self.query_gemini_token_count(prompt)
        elif self.provider == 'vllm':
            # 适配vLLM模型的token计数（需用对应tokenizer）
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')


if __name__ == '__main__':
    def test_model(model_name, prompt):
        print(f'\n{"="*50} Testing model: {model_name} {"="*50}')
        model = Model(model_name)
        print(f'Provider: {model.provider}')
        print(f'Context Limit: {model.context_limit}')
        print(f'Prompt: {prompt}')
        response = model.query(prompt, temperature=0.1)
        print(f'Response: {response}')
        num_tokens = model.get_token_count(prompt)
        print(f'Number of tokens: {num_tokens}')

    # 测试本地模型（核心：使用配置中的别名或完整路径）
    test_prompt = "请介绍一下你自己，以及你擅长的表格推理任务。"
    # 1. 测试Qwen2.5-3B-Instruct（使用别名）
    test_model("qwen_3b", test_prompt)
    # 2. 测试TableLLM-7b（使用别名）
    test_model("tablellm-7b", test_prompt)
    # 3. 也可使用完整tokenizer路径测试（如）
    # test_model("RUCKBReasoning/TableLLM-7b", test_prompt)

    # 保留原有模型测试（可选）
    # test_model("gpt-4o-mini-2024-07-18", "Hello, how are you?")
    # test_model("gemini-1.5-flash", "Hello, how are you?")