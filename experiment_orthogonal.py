# -*- coding: utf-8 -*-
"""
正交验证实验框架
基于单一变量控制原则，验证核心结论
与原实验框架保持一致的日志、输出、格式
"""

import os
import sys
import json
import time
import yaml
import tiktoken
import pandas as pd
import logging
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def setup_logging(result_dir):
    """配置日志系统（与原代码一致）"""
    log_dir = os.path.join(result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def normalize_json_output(output_content):
    """标准化JSON输出（与原代码一致）"""
    try:
        content = output_content.strip()
        
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])
        
        content = content.strip()
        
        if content.startswith('['):
            json_match = re.search(r'\[[\s\S]*\]', content)
        else:
            json_match = re.search(r'\{[\s\S]*\}', content)
        
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            normalized = json.dumps(parsed, ensure_ascii=False, indent=2)
            return normalized, parsed, True
        return output_content, None, False
    except json.JSONDecodeError as e:
        return output_content, None, False


def evaluate_format_compliance(output_content, parsed_json):
    """Python精确判定格式合规情况（与原代码一致）"""
    result = {
        'json_valid': False,
        'json_valid_detail': 'JSON解析失败',
        'structure_valid': False,
        'structure_detail': '结构不完整',
        'field_names_valid': False,
        'field_names_detail': '字段命名不符合要求',
        'data_types_valid': False,
        'data_types_detail': '数据类型不正确'
    }
    
    if parsed_json is None:
        result['json_valid'] = False
        result['json_valid_detail'] = 'JSON解析失败，无法继续检查'
        return result
    
    result['json_valid'] = True
    result['json_valid_detail'] = 'JSON解析成功'
    
    required_fields = ['name', 'methods', 'requirements', 'criteria']
    
    test_items = None
    if isinstance(parsed_json, dict) and 'test_items' in parsed_json:
        test_items = parsed_json['test_items']
        if isinstance(test_items, list) and len(test_items) > 0:
            result['structure_valid'] = True
            result['structure_detail'] = f'结构完整，包含test_items数组，共{len(test_items)}个测试项'
        else:
            result['structure_detail'] = 'test_items不是有效数组或为空'
    elif isinstance(parsed_json, list) and len(parsed_json) > 0:
        test_items = parsed_json
        result['structure_valid'] = False
        result['structure_detail'] = f'输出为数组而非包含test_items字段的对象，共{len(test_items)}个测试项'
    else:
        result['structure_detail'] = '未找到test_items数组或有效测试项'
        return result
    
    if test_items:
        all_fields_correct = True
        all_types_correct = True
        field_issues = []
        type_issues = []
        correct_items = 0
        
        for idx, item in enumerate(test_items):
            if not isinstance(item, dict):
                all_fields_correct = False
                field_issues.append(f'测试项{idx+1}不是对象类型')
                continue
            
            item_fields_ok = True
            for field in required_fields:
                if field not in item:
                    all_fields_correct = False
                    item_fields_ok = False
                    field_issues.append(f'测试项{idx+1}缺少字段"{field}"')
            
            if 'name' in item:
                if not isinstance(item['name'], str):
                    all_types_correct = False
                    type_issues.append(f'测试项{idx+1}的name不是字符串')
            
            for field in ['methods', 'requirements', 'criteria']:
                if field in item:
                    if not isinstance(item[field], list):
                        all_types_correct = False
                        type_issues.append(f'测试项{idx+1}的{field}不是数组')
            
            if item_fields_ok:
                correct_items += 1
        
        if all_fields_correct:
            result['field_names_valid'] = True
            result['field_names_detail'] = f'所有{len(test_items)}个测试项的字段命名完全符合要求'
        else:
            result['field_names_detail'] = f'{correct_items}/{len(test_items)}个测试项字段正确，问题：{"; ".join(field_issues[:3])}'
        
        if all_types_correct:
            result['data_types_valid'] = True
            result['data_types_detail'] = f'所有测试项的数据类型正确'
        else:
            result['data_types_detail'] = f'数据类型问题：{"; ".join(type_issues[:3])}'
    
    return result


class ConfigLoader:
    """配置加载器（正交实验版）"""
    
    def __init__(self, config_path='config_orthogonal.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.scenes = {}
        self._load_scenes()
    
    def _load_scenes(self):
        """加载所有场景配置（与原代码一致）"""
        scenes_dir = 'scenes'
        for scene_file in os.listdir(scenes_dir):
            if scene_file.endswith('.yaml'):
                scene_path = os.path.join(scenes_dir, scene_file)
                with open(scene_path, 'r', encoding='utf-8') as f:
                    scene_id = scene_file.replace('.yaml', '')
                    self.scenes[scene_id] = yaml.safe_load(f)
    
    def get_api_config(self):
        return self.config['api']['siliconflow']
    
    def get_model_config(self, model_key):
        return self.config['models'][model_key]
    
    def get_scenes(self):
        return self.scenes
    
    def get_experiment_config(self):
        return self.config['experiment']
    
    def get_evaluation_dimensions(self):
        return self.config['evaluation_dimensions']


class APIClientManager:
    """API客户端管理器（与原代码一致）"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.clients = {}
        self._init_clients()
    
    def _init_clients(self):
        # Qwen模型使用SiliconFlow API
        siliconflow_api_key = os.getenv('SILICON_FLOW_API_KEY')
        if not siliconflow_api_key:
            raise ValueError("请设置环境变量 SILICON_FLOW_API_KEY")
        
        api_config = self.config_loader.get_api_config()
        for model_key in ['qwen14b', 'qwen32b']:
            self.clients[model_key] = OpenAI(
                api_key=siliconflow_api_key,
                base_url=api_config['base_url']
            )
        
        # DeepSeek使用独立API
        deepseek_api_key = os.getenv('DeepSeek_API_KEY')
        if not deepseek_api_key:
            raise ValueError("请设置环境变量 DeepSeek_API_KEY")
        
        self.clients['deepseek'] = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
    
    def get_client(self, model_key):
        return self.clients.get(model_key)


class TestItemGenerator:
    """测试项生成器（与原代码一致）"""
    
    def __init__(self, config_loader, client_manager, logger):
        self.config_loader = config_loader
        self.client_manager = client_manager
        self.experiment_config = config_loader.get_experiment_config()
        self.logger = logger
    
    def generate(self, scene_config, scheme_key, model_key, temperature, top_p, repeat_idx):
        """生成测试项"""
        start_time = time.time()
        scheme_config = scene_config[scheme_key]
        
        messages = []
        if scheme_config.get('system'):
            messages.append({"role": "system", "content": scheme_config['system']})
        messages.append({"role": "user", "content": scheme_config['user']})
        
        input_prompt = f"{scheme_config.get('system', '')}\n{scheme_config['user']}".strip()
        input_tokens = len(self.config_loader.encoding.encode(input_prompt))
        
        try:
            self.logger.debug(f"调用API: model={model_key}, temp={temperature}, top_p={top_p}")
            output_content = self._call_api(model_key, temperature, top_p, messages)
            output_tokens = len(self.config_loader.encoding.encode(output_content))
            elapsed_time = round(time.time() - start_time, 2)
            
            self.logger.info(f"API调用成功: input_tokens={input_tokens}, output_tokens={output_tokens}, elapsed={elapsed_time}s")
            
            return {
                "success": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_time": elapsed_time,
                "output_content": output_content,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            return {
                "success": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_time": 0,
                "output_content": "",
                "error": str(e)
            }
    
    def _call_api(self, model_key, temperature, top_p, messages):
        """调用API（与原代码一致）"""
        model_config = self.config_loader.get_model_config(model_key)
        client = self.client_manager.get_client(model_key)
        
        max_retries = self.experiment_config['max_retries']
        retry_delay = self.experiment_config['retry_delay']
        
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_config['name'],
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=model_config['max_tokens'],
                    stream=False,
                    timeout=model_config['timeout']
                )
                return response.choices[0].message.content
            except Exception as e:
                if retry < max_retries - 1:
                    self.logger.warning(f"请求超时，{retry_delay}秒后重试... (重试 {retry+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise


class ResultEvaluator:
    """结果评价器（与原代码一致）"""
    
    def __init__(self, config_loader, client_manager, logger):
        self.config_loader = config_loader
        self.client_manager = client_manager
        self.experiment_config = config_loader.get_experiment_config()
        self.dimensions = config_loader.get_evaluation_dimensions()
        self.logger = logger
    
    def evaluate(self, scene_config, output_content, input_prompt):
        """评价生成结果"""
        # 先进行Python格式检查
        normalized_output, parsed_json, is_valid_json = normalize_json_output(output_content)
        format_check = evaluate_format_compliance(normalized_output, parsed_json)
        
        # 构建评价提示词
        eval_prompt = scene_config['evaluation_prompt'].format(
            input_prompt=input_prompt,
            output_content=normalized_output,
            json_valid=format_check['json_valid'],
            json_valid_detail=format_check['json_valid_detail'],
            structure_valid=format_check['structure_valid'],
            structure_detail=format_check['structure_detail'],
            field_names_valid=format_check['field_names_valid'],
            field_names_detail=format_check['field_names_detail'],
            data_types_valid=format_check['data_types_valid'],
            data_types_detail=format_check['data_types_detail']
        )
        
        try:
            self.logger.debug("调用评价API...")
            eval_result = self._call_eval_api(eval_prompt)
            self.logger.info(f"评价完成: {eval_result}")
            return {
                "success": True,
                **eval_result,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"评价失败: {str(e)}")
            return {
                "success": False,
                **{dim['name']: 0 for dim in self.dimensions},
                "error": str(e)
            }
    
    def _call_eval_api(self, eval_prompt):
        """调用评价API（与原代码一致）"""
        client = self.client_manager.get_client('deepseek')
        max_retries = self.experiment_config['max_retries']
        retry_delay = self.experiment_config['retry_delay']
        
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0,
                    timeout=60
                )
                content = response.choices[0].message.content
                
                # 提取JSON
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
                return json.loads(content)
            except Exception as e:
                if retry < max_retries - 1:
                    self.logger.warning(f"评价请求超时，{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise


class OrthogonalExperiment:
    """正交验证实验主类（支持断点续跑）"""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.client_manager = APIClientManager(self.config_loader)
        
        # 创建结果目录
        self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       'experiment_results_orthogonal')
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'raw_outputs'), exist_ok=True)
        
        # 设置日志
        self.logger = setup_logging(self.result_dir)
        
        self.generator = TestItemGenerator(self.config_loader, self.client_manager, self.logger)
        self.evaluator = ResultEvaluator(self.config_loader, self.client_manager, self.logger)
        
        self.results = []
        self.exp_config = self.config_loader.get_experiment_config()
        
        # 加载已完成的实验记录（用于断点续跑）
        self.completed_experiments = self._load_completed_experiments()
        self.skipped_count = 0
    
    def _get_experiment_key(self, group_id, scene_id, scheme_key, model_key, temperature, repeat_idx):
        """生成实验唯一标识键"""
        model_name = self.config_loader.get_model_config(model_key)['name']
        return f"{group_id}_{scene_id}_{scheme_key}_T{temperature}_{model_name}_{repeat_idx+1}"
    
    def _load_completed_experiments(self):
        """加载已完成的实验记录"""
        completed = set()
        raw_outputs_dir = os.path.join(self.result_dir, 'raw_outputs')
        
        if not os.path.exists(raw_outputs_dir):
            return completed
        
        # 从已有的JSON文件加载
        for filename in os.listdir(raw_outputs_dir):
            if filename.endswith('.json'):
                # 解析文件名: group_scene_scheme_Ttemp_model_repeat.json
                try:
                    parts = filename.replace('.json', '').split('_')
                    if len(parts) >= 6:
                        key = filename.replace('.json', '')
                        completed.add(key)
                except:
                    continue
        
        # 同时从CSV加载（如果存在）
        csv_path = os.path.join(self.result_dir, 'results.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    key = f"{row['group']}_{row['scene_id']}_{row['scheme']}_T{row['temperature']}_{row['model']}_{row['repeat_idx']}"
                    completed.add(key)
            except:
                pass
        
        if completed:
            self.logger.info(f"检测到 {len(completed)} 个已完成实验，将跳过这些实验")
        
        return completed
    
    def _is_experiment_completed(self, group_id, scene_id, scheme_key, model_key, temperature, repeat_idx):
        """检查实验是否已完成"""
        key = self._get_experiment_key(group_id, scene_id, scheme_key, model_key, temperature, repeat_idx)
        return key in self.completed_experiments
    
    def _run_single_experiment(self, scene_config, scene_id, scheme_key,
                                model_key, temperature, top_p, repeat_idx, group_id):
        """运行单次实验（与原代码一致）"""
        model_config = self.config_loader.get_model_config(model_key)
        scheme_config = scene_config[scheme_key]
        input_prompt = f"{scheme_config.get('system', '')}\n{scheme_config['user']}".strip()
        
        # 生成
        gen_result = self.generator.generate(
            scene_config, scheme_key, model_key, temperature, top_p, repeat_idx
        )
        
        if not gen_result['success']:
            self.logger.error(f"生成失败: {gen_result['error']}")
            return self._create_failed_result(
                group_id, scene_id, scheme_key, model_config['name'],
                temperature, repeat_idx, gen_result['error']
            )
        
        # 评价
        eval_result = self.evaluator.evaluate(
            scene_config, gen_result['output_content'], input_prompt
        )
        
        # 构建结果（与原代码格式一致）
        result = {
            'group': group_id,
            'scene_id': scene_id,
            'scene_name': scene_config['scene_name'],
            'scheme': scheme_config['name'],
            'model': model_config['name'],
            'temperature': temperature,
            'top_p': top_p,
            'repeat_idx': repeat_idx + 1,
            'input_tokens': gen_result['input_tokens'],
            'output_tokens': gen_result['output_tokens'],
            'elapsed_time': gen_result['elapsed_time'],
            'output_content': gen_result['output_content'],
            'success': True,
            'error': None
        }
        
        for dim in self.config_loader.get_evaluation_dimensions():
            result[dim['name']] = eval_result.get(dim['name'], 0)
        
        self.logger.info(f"完成: input_tokens={result['input_tokens']}, "
                        f"output_tokens={result['output_tokens']}, "
                        f"elapsed_time={result['elapsed_time']}s | "
                        f"格式合规率={result['格式合规率']}, "
                        f"条目对应率={result['条目对应率']}, "
                        f"需求覆盖率={result['需求覆盖率']}, "
                        f"测试项有效性={result.get('测试项有效性', 'N/A')}, "
                        f"输出描述质量={result.get('输出描述质量', 'N/A')}, "
                        f"无幻觉率={result['无幻觉率']}")
        
        return result
    
    def _create_failed_result(self, group_id, scene_id, scheme_key, model_name,
                              temperature, repeat_idx, error):
        """创建失败结果（与原代码一致）"""
        return {
            'group': group_id,
            'scene_id': scene_id,
            'scheme': scheme_key,
            'model': model_name,
            'temperature': temperature,
            'repeat_idx': repeat_idx + 1,
            'success': False,
            'error': error,
            **{dim['name']: 0 for dim in self.config_loader.get_evaluation_dimensions()}
        }
    
    def _save_raw_output(self, result):
        """保存原始输出（与原代码一致）"""
        filename = (f"{result['group']}_{result['scene_id']}_{result['scheme']}"
                   f"_T{result['temperature']}_{result['model'].replace('/', '_')}_{result['repeat_idx']}.json")
        filepath = os.path.join(self.result_dir, 'raw_outputs', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def run_group1_temperature(self):
        """实验组1：温度影响"""
        self.logger.info("="*60)
        self.logger.info("========== 实验组1：验证温度影响（单因素实验） ==========")
        self.logger.info("="*60)
        
        config = self.exp_config['group1_temperature']
        scene_config = self.config_loader.get_scenes()['sceneA']
        scheme_key = config['scheme']
        
        total = len(config['models']) * len(config['temperatures']) * config['sample_size']
        current = 0
        
        for model_key in config['models']:
            for temp in config['temperatures']:
                for i in range(config['sample_size']):
                    current += 1
                    
                    # 检查是否已完成
                    if self._is_experiment_completed('group1', 'sceneA', scheme_key, model_key, temp, i):
                        self.logger.info(f"[{current}/{total}] 跳过（已完成）: 场景A-测控指令解码 | "
                                       f"{self.config_loader.get_model_config(model_key)['name']} | "
                                       f"方案2-结构化上下文 | T{temp} | 第{i+1}次")
                        self.skipped_count += 1
                        continue
                    
                    self.logger.info(f"[{current}/{total}] 场景A-测控指令解码 | "
                                   f"{self.config_loader.get_model_config(model_key)['name']} | "
                                   f"方案2-结构化上下文 | T{temp} | 第{i+1}次")
                    
                    result = self._run_single_experiment(
                        scene_config, 'sceneA', scheme_key,
                        model_key, temp, 1.0, i, 'group1'
                    )
                    self.results.append(result)
                    self._save_raw_output(result)
    
    def run_group2_scheme(self):
        """实验组2：方案影响"""
        self.logger.info("="*60)
        self.logger.info("========== 实验组2：验证方案影响（单因素实验） ==========")
        self.logger.info("="*60)
        
        config = self.exp_config['group2_scheme']
        scene_config = self.config_loader.get_scenes()['sceneA']
        temp = config['temperature']
        
        total = len(config['models']) * len(config['schemes']) * config['sample_size']
        current = 0
        
        for model_key in config['models']:
            for scheme_key in config['schemes']:
                scheme_name = scene_config[scheme_key]['name']
                for i in range(config['sample_size']):
                    current += 1
                    
                    # 检查是否已完成
                    if self._is_experiment_completed('group2', 'sceneA', scheme_key, model_key, temp, i):
                        self.logger.info(f"[{current}/{total}] 跳过（已完成）: 场景A-测控指令解码 | "
                                       f"{self.config_loader.get_model_config(model_key)['name']} | "
                                       f"{scheme_name} | T{temp} | 第{i+1}次")
                        self.skipped_count += 1
                        continue
                    
                    self.logger.info(f"[{current}/{total}] 场景A-测控指令解码 | "
                                   f"{self.config_loader.get_model_config(model_key)['name']} | "
                                   f"{scheme_name} | T{temp} | 第{i+1}次")
                    
                    result = self._run_single_experiment(
                        scene_config, 'sceneA', scheme_key,
                        model_key, temp, 1.0, i, 'group2'
                    )
                    self.results.append(result)
                    self._save_raw_output(result)
    
    def run_group3_model(self):
        """实验组3：模型影响"""
        self.logger.info("="*60)
        self.logger.info("========== 实验组3：验证模型影响（单因素实验） ==========")
        self.logger.info("="*60)
        
        config = self.exp_config['group3_model']
        scene_config = self.config_loader.get_scenes()['sceneA']
        temp = config['temperature']
        
        total = len(config['models']) * len(config['schemes']) * config['sample_size']
        current = 0
        
        for model_key in config['models']:
            for scheme_key in config['schemes']:
                scheme_name = scene_config[scheme_key]['name']
                for i in range(config['sample_size']):
                    current += 1
                    
                    # 检查是否已完成
                    if self._is_experiment_completed('group3', 'sceneA', scheme_key, model_key, temp, i):
                        self.logger.info(f"[{current}/{total}] 跳过（已完成）: 场景A-测控指令解码 | "
                                       f"{self.config_loader.get_model_config(model_key)['name']} | "
                                       f"{scheme_name} | T{temp} | 第{i+1}次")
                        self.skipped_count += 1
                        continue
                    
                    self.logger.info(f"[{current}/{total}] 场景A-测控指令解码 | "
                                   f"{self.config_loader.get_model_config(model_key)['name']} | "
                                   f"{scheme_name} | T{temp} | 第{i+1}次")
                    
                    result = self._run_single_experiment(
                        scene_config, 'sceneA', scheme_key,
                        model_key, temp, 1.0, i, 'group3'
                    )
                    self.results.append(result)
                    self._save_raw_output(result)
    
    def run_group4_interaction(self):
        """实验组4：交互效应"""
        self.logger.info("="*60)
        self.logger.info("========== 实验组4：验证交互效应（正交设计） ==========")
        self.logger.info("="*60)
        
        config = self.exp_config['group4_interaction']
        scene_config = self.config_loader.get_scenes()['sceneA']
        
        total = len(config['combinations']) * config['sample_size']
        current = 0
        
        for combo in config['combinations']:
            scheme_name = scene_config[combo['scheme']]['name']
            for i in range(config['sample_size']):
                current += 1
                
                # 检查是否已完成
                if self._is_experiment_completed('group4', 'sceneA', combo['scheme'], combo['model'], combo['temp'], i):
                    self.logger.info(f"[{current}/{total}] 跳过（已完成）: 场景A-测控指令解码 | "
                                   f"{self.config_loader.get_model_config(combo['model'])['name']} | "
                                   f"{scheme_name} | T{combo['temp']} | 第{i+1}次")
                    self.skipped_count += 1
                    continue
                
                self.logger.info(f"[{current}/{total}] 场景A-测控指令解码 | "
                               f"{self.config_loader.get_model_config(combo['model'])['name']} | "
                               f"{scheme_name} | T{combo['temp']} | 第{i+1}次")
                
                result = self._run_single_experiment(
                    scene_config, 'sceneA', combo['scheme'],
                    combo['model'], combo['temp'], 1.0, i, 'group4'
                )
                self.results.append(result)
                self._save_raw_output(result)
    
    def _generate_reports(self):
        """生成报告（与原代码一致）"""
        if not self.results:
            self.logger.warning("没有实验结果")
            return
        
        df = pd.DataFrame(self.results)
        
        # 保存CSV
        csv_path = os.path.join(self.result_dir, 'results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 生成统计报告
        self._generate_statistics(df)
        
        self.logger.info(f"报告已保存到: {self.result_dir}")
    
    def _generate_statistics(self, df):
        """生成统计报告"""
        stats = []
        
        # 按实验组统计
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            
            for model in group_df['model'].unique():
                model_df = group_df[group_df['model'] == model]
                
                row = {
                    'group': group,
                    'model': model,
                    'n': len(model_df),
                    'mean_score': model_df[['格式合规率', '条目对应率', '需求覆盖率',
                                           '测试项有效性', '输出描述质量', '无幻觉率']].mean().mean(),
                    'std_score': model_df[['格式合规率', '条目对应率', '需求覆盖率',
                                          '测试项有效性', '输出描述质量', '无幻觉率']].mean().std()
                }
                stats.append(row)
        
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(self.result_dir, 'statistics.csv')
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    
    def run(self):
        """运行所有实验组"""
        self.logger.info("="*60)
        self.logger.info("正交验证实验自动化框架")
        self.logger.info("="*60)
        
        start_time = datetime.now()
        
        # 实验组1：温度影响
        if self.exp_config['group1_temperature']['enabled']:
            self.run_group1_temperature()
        
        # 实验组2：方案影响
        if self.exp_config['group2_scheme']['enabled']:
            self.run_group2_scheme()
        
        # 实验组3：模型影响
        if self.exp_config['group3_model']['enabled']:
            self.run_group3_model()
        
        # 实验组4：交互效应
        if self.exp_config['group4_interaction']['enabled']:
            self.run_group4_interaction()
        
        # 生成报告
        self._generate_reports()
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        self.logger.info("="*60)
        self.logger.info(f"实验完成！总耗时: {elapsed}")
        self.logger.info(f"本次新完成实验: {len(self.results)}")
        if self.skipped_count > 0:
            self.logger.info(f"跳过已完成的实验: {self.skipped_count}")
        self.logger.info(f"累计完成实验: {len(self.results) + self.skipped_count}")
        self.logger.info("="*60)


def main():
    """主函数"""
    experiment = OrthogonalExperiment()
    experiment.run()


if __name__ == '__main__':
    main()
