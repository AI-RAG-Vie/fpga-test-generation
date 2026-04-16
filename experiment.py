# -*- coding: utf-8 -*-
"""
论文实验自动化框架
支持多场景、多方案、多参数、多模型的自动化实验
支持中断继续和详细日志记录
支持JSON格式标准化和多次实验一致性评价
"""

import os
import time
import json
import yaml
import tiktoken
import pandas as pd
import logging
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def setup_logging(result_dir):
    """配置日志系统"""
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
    """标准化JSON输出，确保格式一致"""
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
    """
    Python精确判定格式合规情况，收集判定信息供DeepSeek打分
    
    检查项:
    - JSON有效性：能被JSON解析器正确解析
    - 结构完整性：包含test_items数组，每个item包含name/methods/requirements/criteria
    - 字段命名规范：字段名称完全符合要求
    - 数据类型正确：name为字符串，methods/requirements/criteria为数组
    
    返回: 判定详情字典
    """
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


def calculate_consistency(results_group):
    """计算多次实验结果的一致性"""
    if len(results_group) < 2:
        return 100.0
    
    scores = []
    for dim in ['格式合规率', '条目对应率', '需求覆盖率', '测试项有效性', '输出描述质量', '无幻觉率']:
        values = [r.get(dim, 0) for r in results_group if dim in r]
        if values:
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std_dev = variance ** 0.5
            cv = (std_dev / mean_val * 100) if mean_val > 0 else 0
            scores.append(100 - min(cv, 100))
    
    return sum(scores) / len(scores) if scores else 100.0


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.scenes = {}
        self._load_scenes()
    
    def _load_scenes(self):
        """加载所有场景配置"""
        scenes_dir = self.config['experiment']['scenes_dir']
        for scene_file in self.config['experiment']['scenes']:
            scene_path = os.path.join(scenes_dir, scene_file)
            with open(scene_path, 'r', encoding='utf-8') as f:
                scene_config = yaml.safe_load(f)
                self.scenes[scene_config['scene_id']] = scene_config
    
    def get_model_config(self, model_key):
        return self.config['models'][model_key]
    
    def get_parameter_sets(self):
        return self.config['parameter_sets']
    
    def get_experiment_config(self):
        return self.config['experiment']
    
    def get_scenes(self):
        return self.scenes
    
    def get_evaluation_dimensions(self):
        return self.config['evaluation_dimensions']


class ProgressTracker:
    """进度跟踪器，支持中断继续"""
    
    def __init__(self, result_dir):
        self.progress_file = os.path.join(result_dir, 'progress.json')
        self.completed = set()
        self._load_progress()
    
    def _load_progress(self):
        """加载已有进度"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.completed = set(data.get('completed', []))
    
    def _save_progress(self):
        """保存进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump({'completed': list(self.completed)}, f)
    
    def is_completed(self, key):
        """检查是否已完成"""
        return key in self.completed
    
    def mark_completed(self, key):
        """标记为已完成"""
        self.completed.add(key)
        self._save_progress()
    
    def get_completed_count(self):
        """获取已完成数量"""
        return len(self.completed)


class APIClientManager:
    """API客户端管理器"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.clients = {}
    
    def get_client(self, model_key):
        if model_key not in self.clients:
            model_config = self.config_loader.get_model_config(model_key)
            self.clients[model_key] = OpenAI(
                api_key=os.getenv(model_config['api_key_env']),
                base_url=model_config['base_url']
            )
        return self.clients[model_key]


class TestItemGenerator:
    """测试项生成器"""
    
    def __init__(self, config_loader, client_manager, logger):
        self.config_loader = config_loader
        self.client_manager = client_manager
        self.experiment_config = config_loader.get_experiment_config()
        self.logger = logger
    
    def generate(self, scene_config, scheme_key, model_key, param_set, repeat_idx):
        start_time = time.time()
        scheme_config = scene_config[scheme_key]
        
        messages = []
        if scheme_config['system']:
            messages.append({"role": "system", "content": scheme_config['system']})
        messages.append({"role": "user", "content": scheme_config['user']})
        
        input_tokens = len(self.config_loader.encoding.encode(
            scheme_config['system'] + scheme_config['user']
        ))
        
        try:
            self.logger.debug(f"调用API: model={model_key}, temp={param_set['temperature']}, top_p={param_set['top_p']}")
            output_content = self._call_api(model_key, param_set, messages, scheme_key, repeat_idx)
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
    
    def _call_api(self, model_key, param_set, messages, scheme_key, repeat_idx):
        model_config = self.config_loader.get_model_config(model_key)
        client = self.client_manager.get_client(model_key)
        
        max_retries = self.experiment_config['max_retries']
        retry_delay = self.experiment_config['retry_delay']
        
        print("\n" + "="*80)
        print(f"[终端调试] 生成API请求 | 方案:{scheme_key} | 第{repeat_idx+1}次")
        print(f"[终端调试] 模型: {model_config['model_name']}")
        print(f"[终端调试] 参数: temperature={param_set['temperature']}, top_p={param_set['top_p']}")
        print(f"[终端调试] 消息内容:")
        for msg in messages:
            print(f"  [{msg['role']}]: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"  [{msg['role']}]: {msg['content']}")
        
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=messages,
                    temperature=param_set['temperature'],
                    top_p=param_set['top_p'],
                    max_tokens=model_config['max_tokens'],
                    stream=False,
                    timeout=model_config['timeout']
                )
                content = response.choices[0].message.content
                
                print(f"\n[终端调试] 生成API响应:")
                print(f"[终端调试] 响应长度: {len(content)} 字符")
                print(f"[终端调试] 响应内容预览: {content[:500]}..." if len(content) > 500 else f"[终端调试] 响应内容: {content}")
                print("="*80 + "\n")
                
                return content
            except Exception as e:
                if retry < max_retries - 1:
                    self.logger.warning(f"请求超时，{retry_delay}秒后重试... (重试 {retry+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"\n[终端调试] API调用失败: {str(e)}")
                    print("="*80 + "\n")
                    raise


class ResultEvaluator:
    """结果评价器"""
    
    def __init__(self, config_loader, client_manager, logger):
        self.config_loader = config_loader
        self.client_manager = client_manager
        self.experiment_config = config_loader.get_experiment_config()
        self.dimensions = config_loader.get_evaluation_dimensions()
        self.logger = logger
    
    def evaluate(self, scene_config, output_content, input_prompt, format_check):
        eval_prompt = scene_config['evaluation_prompt'].format(
            input_prompt=input_prompt,
            output_content=output_content,
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
        model_config = self.config_loader.get_model_config('deepseek')
        client = self.client_manager.get_client('deepseek')
        
        max_retries = self.experiment_config['max_retries']
        retry_delay = self.experiment_config['retry_delay']
        
        print("\n" + "-"*80)
        print("[终端调试] 评价API请求")
        print(f"[终端调试] 模型: {model_config['model_name']}")
        print(f"[终端调试] 评价提示词预览: {eval_prompt[:300]}..." if len(eval_prompt) > 300 else f"[终端调试] 评价提示词: {eval_prompt}")
        
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=model_config['temperature'],
                    timeout=model_config['timeout']
                )
                content = response.choices[0].message.content
                
                print(f"\n[终端调试] 评价API响应:")
                print(f"[终端调试] 响应长度: {len(content)} 字符")
                print(f"[终端调试] 响应内容: {content}")
                print("-"*80 + "\n")
                
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
                    print(f"\n[终端调试] 评价API调用失败: {str(e)}")
                    print("-"*80 + "\n")
                    raise


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.experiment_config = self.config_loader.get_experiment_config()
        
        result_dir = self.experiment_config['result_dir']
        self.logger = setup_logging(result_dir)
        
        self.client_manager = APIClientManager(self.config_loader)
        self.generator = TestItemGenerator(self.config_loader, self.client_manager, self.logger)
        self.evaluator = ResultEvaluator(self.config_loader, self.client_manager, self.logger)
        self.progress_tracker = ProgressTracker(result_dir)
        
        self.results = []
        self._setup_directories()
        self._load_existing_results()
    
    def _setup_directories(self):
        result_dir = self.experiment_config['result_dir']
        os.makedirs(os.path.join(result_dir, 'raw_outputs'), exist_ok=True)
        os.makedirs(os.path.join(result_dir, 'scores'), exist_ok=True)
        os.makedirs(os.path.join(result_dir, 'summary'), exist_ok=True)
    
    def _load_existing_results(self):
        """加载已有结果"""
        scores_path = os.path.join(self.experiment_config['result_dir'], 'scores', 'all_scores.csv')
        if os.path.exists(scores_path):
            df = pd.read_csv(scores_path)
            self.results = df.to_dict('records')
            self.logger.info(f"加载已有结果: {len(self.results)}条")
    
    def run(self):
        """运行完整实验 - 先完成一个模型的所有实验，再切换模型"""
        scenes = self.config_loader.get_scenes()
        param_sets = self.config_loader.get_parameter_sets()
        repeat_times = self.experiment_config['repeat_times']
        
        main_exp = self.experiment_config['main_experiment']
        sens_exp = self.experiment_config['sensitivity_experiment']
        
        total_main = len(main_exp['scenes']) * len(main_exp['schemes']) * len(main_exp['models']) * repeat_times
        total_sens = len(sens_exp['schemes']) * len(sens_exp['param_sets']) * len(sens_exp['models']) * repeat_times
        total_experiments = total_main + total_sens
        
        self.logger.info("="*60)
        self.logger.info("论文实验自动化框架")
        self.logger.info(f"主实验: {total_main}次 ({len(main_exp['scenes'])}场景 × {len(main_exp['schemes'])}方案 × {len(main_exp['models'])}模型 × {repeat_times}次)")
        self.logger.info(f"参数敏感性实验: {total_sens}次 ({len(sens_exp['schemes'])}方案 × {len(sens_exp['param_sets'])}参数 × {len(sens_exp['models'])}模型 × {repeat_times}次)")
        self.logger.info(f"总计: {total_experiments}次")
        self.logger.info(f"已完成: {self.progress_tracker.get_completed_count()}次")
        self.logger.info("="*60)
        
        start_time = datetime.now()
        current = 0
        
        if main_exp['enabled']:
            self.logger.info("\n========== 开始主实验 ==========")
            for model_key in main_exp['models']:
                model_config = self.config_loader.get_model_config(model_key)
                self.logger.info(f"\n----- 切换模型: {model_config['name']} -----")
                
                for scene_id in main_exp['scenes']:
                    scene_config = scenes[scene_id]
                    for scheme_key in main_exp['schemes']:
                        for repeat_idx in range(repeat_times):
                            current += 1
                            exp_key = f"main_{scene_id}_{scheme_key}_{model_key}_P1_{repeat_idx}"
                            
                            if self.progress_tracker.is_completed(exp_key):
                                self.logger.info(f"[{current}/{total_experiments}] 已完成，跳过")
                                continue
                            
                            scheme_name = scene_config[scheme_key]['name']
                            param_set = param_sets['P1']
                            
                            self.logger.info(f"[{current}/{total_experiments}] "
                                           f"{scene_config['scene_name']} | "
                                           f"{model_config['name']} | "
                                           f"{scheme_name} | P1 | 第{repeat_idx+1}次")
                            
                            result = self._run_single_experiment(
                                scene_config, scene_id, scheme_key,
                                model_key, 'P1', param_set, repeat_idx
                            )
                            self.results.append(result)
                            self.progress_tracker.mark_completed(exp_key)
        
        if sens_exp['enabled']:
            self.logger.info("\n========== 开始参数敏感性实验 ==========")
            scene_config = scenes[sens_exp['scene']]
            
            for model_key in sens_exp['models']:
                model_config = self.config_loader.get_model_config(model_key)
                self.logger.info(f"\n----- 切换模型: {model_config['name']} -----")
                
                for scheme_key in sens_exp['schemes']:
                    for param_key in sens_exp['param_sets']:
                        for repeat_idx in range(repeat_times):
                            current += 1
                            exp_key = f"sens_{sens_exp['scene']}_{scheme_key}_{model_key}_{param_key}_{repeat_idx}"
                            
                            if self.progress_tracker.is_completed(exp_key):
                                self.logger.info(f"[{current}/{total_experiments}] 已完成，跳过")
                                continue
                            
                            scheme_name = scene_config[scheme_key]['name']
                            param_set = param_sets[param_key]
                            
                            self.logger.info(f"[{current}/{total_experiments}] "
                                           f"{scene_config['scene_name']} | "
                                           f"{model_config['name']} | "
                                           f"{scheme_name} | {param_key} | 第{repeat_idx+1}次")
                            
                            result = self._run_single_experiment(
                                scene_config, sens_exp['scene'], scheme_key,
                                model_key, param_key, param_set, repeat_idx
                            )
                            self.results.append(result)
                            self.progress_tracker.mark_completed(exp_key)
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"实验完成！总耗时: {elapsed}")
        self.logger.info("="*60)
        
        self._generate_reports()
    
    def _run_single_experiment(self, scene_config, scene_id, scheme_key,
                                model_key, param_key, param_set, repeat_idx):
        model_config = self.config_loader.get_model_config(model_key)
        scheme_name = scene_config[scheme_key]['name']
        scheme_config = scene_config[scheme_key]
        input_prompt = f"{scheme_config.get('system', '')}\n{scheme_config.get('user', '')}".strip()
        
        gen_result = self.generator.generate(
            scene_config, scheme_key, model_key, param_set, repeat_idx
        )
        
        if not gen_result['success']:
            self.logger.error(f"生成失败: {gen_result['error']}")
            return self._create_failed_result(
                scene_id, scene_config['scene_name'], scheme_name,
                model_config['name'], param_key, repeat_idx, gen_result['error']
            )
        
        normalized_output, parsed_json, is_valid_json = normalize_json_output(gen_result['output_content'])
        
        format_check = evaluate_format_compliance(normalized_output, parsed_json)
        
        self.logger.info(f"Python格式检查: JSON有效={format_check['json_valid']}, "
                        f"结构完整={format_check['structure_valid']}, "
                        f"字段规范={format_check['field_names_valid']}, "
                        f"类型正确={format_check['data_types_valid']}")
        
        eval_result = self.evaluator.evaluate(
            scene_config, normalized_output, input_prompt, format_check
        )
        
        result = {
            'scene_id': scene_id,
            'scene_name': scene_config['scene_name'],
            'scheme': scheme_name,
            'model': model_config['name'],
            'param_set': param_key,
            'temperature': param_set['temperature'],
            'top_p': param_set['top_p'],
            'repeat_idx': repeat_idx + 1,
            'input_tokens': gen_result['input_tokens'],
            'output_tokens': gen_result['output_tokens'],
            'elapsed_time': gen_result['elapsed_time'],
            'output_content': normalized_output,
            'json_valid': is_valid_json,
            'success': True,
            'error': None
        }
        
        for dim in self.config_loader.get_evaluation_dimensions():
            result[dim['name']] = eval_result.get(dim['name'], 0)
        
        if '评分说明' in eval_result:
            result['评分说明'] = eval_result['评分说明']
        
        self._save_raw_output(result)
        
        self.logger.info(f"完成: input_tokens={gen_result['input_tokens']}, "
                        f"output_tokens={gen_result['output_tokens']}, "
                        f"elapsed_time={gen_result['elapsed_time']}s | "
                        f"格式合规率={result['格式合规率']}, "
                        f"条目对应率={result['条目对应率']}, "
                        f"需求覆盖率={result['需求覆盖率']}, "
                        f"测试项有效性={result.get('测试项有效性', 'N/A')}, "
                        f"输出描述质量={result.get('输出描述质量', 'N/A')}, "
                        f"无幻觉率={result['无幻觉率']}")
        
        return result
    
    def _create_failed_result(self, scene_id, scene_name, scheme, model,
                              param_set, repeat_idx, error):
        result = {
            'scene_id': scene_id,
            'scene_name': scene_name,
            'scheme': scheme,
            'model': model,
            'param_set': param_set,
            'repeat_idx': repeat_idx + 1,
            'input_tokens': 0,
            'output_tokens': 0,
            'elapsed_time': 0,
            'output_content': '',
            'success': False,
            'error': error
        }
        
        for dim in self.config_loader.get_evaluation_dimensions():
            result[dim['name']] = 0
        
        return result
    
    def _save_raw_output(self, result):
        filename = f"{result['scene_id']}_{result['scheme']}_{result['param_set']}_{result['model']}_{result['repeat_idx']}.json"
        filename = filename.replace(' ', '_').replace('-', '_')
        
        result_dir = self.experiment_config['result_dir']
        filepath = os.path.join(result_dir, 'raw_outputs', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def _generate_reports(self):
        result_dir = self.experiment_config['result_dir']
        dimensions = self.config_loader.get_evaluation_dimensions()
        
        df = pd.DataFrame(self.results)
        
        scores_path = os.path.join(result_dir, 'scores', 'all_scores.csv')
        df.to_csv(scores_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"评分记录已保存: {scores_path}")
        
        self._generate_consistency_report(df, result_dir, dimensions)
        
        self._generate_summary_reports(df, result_dir, dimensions)
        self._generate_final_report(df, result_dir, dimensions)
    
    def _generate_consistency_report(self, df, result_dir, dimensions):
        """生成一致性评价报告"""
        consistency_data = []
        
        group_cols = ['scene_name', 'model', 'scheme', 'param_set']
        
        for name, group in df.groupby(group_cols):
            if len(group) > 1:
                consistency_score = calculate_consistency(group.to_dict('records'))
                
                row = {
                    '场景': name[0],
                    '模型': name[1],
                    '方案': name[2],
                    '参数组合': name[3],
                    '样本数': len(group),
                    '一致性得分': round(consistency_score, 1)
                }
                
                for dim in dimensions:
                    values = group[dim['name']].tolist()
                    row[f"{dim['name']}_均值"] = round(sum(values) / len(values), 1)
                    row[f"{dim['name']}_标准差"] = round(pd.Series(values).std(), 1)
                    row[f"{dim['name']}_变异系数"] = round(pd.Series(values).std() / (sum(values) / len(values)) * 100, 1) if sum(values) > 0 else 0
                
                consistency_data.append(row)
        
        if consistency_data:
            consistency_df = pd.DataFrame(consistency_data)
            consistency_path = os.path.join(result_dir, 'summary', 'consistency_report.csv')
            consistency_df.to_csv(consistency_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"一致性报告已保存: {consistency_path}")
    
    def _generate_summary_reports(self, df, result_dir, dimensions):
        group_cols = ['scene_name', 'model', 'scheme', 'param_set']
        
        agg_dict = {}
        for dim in dimensions:
            agg_dict[dim['name']] = ['mean', 'std']
        agg_dict['input_tokens'] = ['mean', 'std']
        agg_dict['output_tokens'] = ['mean', 'std']
        agg_dict['elapsed_time'] = ['mean', 'std']
        
        summary = df.groupby(group_cols).agg(agg_dict).round(2)
        
        summary_path = os.path.join(result_dir, 'summary', 'summary_statistics.csv')
        summary.to_csv(summary_path, encoding='utf-8-sig')
        self.logger.info(f"统计汇总已保存: {summary_path}")
        
        main_df = df[df['param_set'] == 'P1'].copy()
        
        main_group_cols = ['scene_name', 'model', 'scheme']
        main_agg_dict = {}
        for dim in dimensions:
            main_agg_dict[dim['name']] = ['mean', 'std']
        main_agg_dict['input_tokens'] = ['mean']
        main_agg_dict['output_tokens'] = ['mean']
        main_agg_dict['elapsed_time'] = ['mean']
        
        main_summary = main_df.groupby(main_group_cols).agg(main_agg_dict).round(2)
        
        main_path = os.path.join(result_dir, 'summary', 'main_results.csv')
        main_summary.to_csv(main_path, encoding='utf-8-sig')
        self.logger.info(f"主实验结果已保存: {main_path}")
    
    def _generate_final_report(self, df, result_dir, dimensions):
        report_path = os.path.join(result_dir, 'final_report.xlsx')
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            detail_cols = ['scene_name', 'model', 'scheme', 'param_set', 'temperature', 'top_p',
                          'repeat_idx', 'input_tokens', 'output_tokens', 'elapsed_time',
                          '格式合规率', '条目对应率', '需求覆盖率', '测试项有效性', '输出描述质量', '无幻觉率',
                          '格式合规率说明', '条目对应率说明', '需求覆盖率说明', '测试项有效性说明', 
                          '输出描述质量说明', '无幻觉率说明']
            detail_cols = [c for c in detail_cols if c in df.columns]
            df[detail_cols].to_excel(writer, sheet_name='详细数据', index=False)
            
            for scene_name in df['scene_name'].unique():
                scene_df = df[df['scene_name'] == scene_name]
                sheet_name = scene_name[:20].replace('/', '_')
                scene_df[detail_cols].to_excel(writer, sheet_name=sheet_name, index=False)
            
            p1_df = df[df['param_set'] == 'P1']
            summary_data = []
            for scene in p1_df['scene_name'].unique():
                for model in p1_df['model'].unique():
                    for scheme in p1_df['scheme'].unique():
                        subset = p1_df[(p1_df['scene_name'] == scene) &
                                      (p1_df['model'] == model) &
                                      (p1_df['scheme'] == scheme)]
                        if len(subset) > 0:
                            row = {
                                '场景': scene,
                                '模型': model,
                                '方案': scheme,
                                '温度': subset['temperature'].iloc[0],
                                'Top_p': subset['top_p'].iloc[0],
                                '样本数': len(subset)
                            }
                            for dim in dimensions:
                                row[f"{dim['name']}_均值"] = round(subset[dim['name']].mean(), 1)
                                row[f"{dim['name']}_标准差"] = round(subset[dim['name']].std(), 1)
                            row['输入Token均值'] = round(subset['input_tokens'].mean())
                            row['输出Token均值'] = round(subset['output_tokens'].mean())
                            row['耗时均值(s)'] = round(subset['elapsed_time'].mean(), 1)
                            summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='主实验汇总', index=False)
            
            sensitivity_data = []
            for model in df['model'].unique():
                for param in df['param_set'].unique():
                    param_df = df[(df['param_set'] == param) & (df['model'] == model)]
                    if len(param_df) > 0:
                        row = {
                            '模型': model,
                            '参数组合': param,
                            '温度': param_df['temperature'].iloc[0],
                            'Top_p': param_df['top_p'].iloc[0],
                            '样本数': len(param_df)
                        }
                        for dim in dimensions:
                            row[f"{dim['name']}_均值"] = round(param_df[dim['name']].mean(), 1)
                            row[f"{dim['name']}_标准差"] = round(param_df[dim['name']].std(), 1)
                        row['输入Token均值'] = round(param_df['input_tokens'].mean())
                        row['输出Token均值'] = round(param_df['output_tokens'].mean())
                        row['耗时均值(s)'] = round(param_df['elapsed_time'].mean(), 1)
                        sensitivity_data.append(row)
            
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df.to_excel(writer, sheet_name='参数敏感性', index=False)
            
            overall_data = []
            for model in df['model'].unique():
                for scene in df['scene_name'].unique():
                    for scheme in df['scheme'].unique():
                        subset = df[(df['model'] == model) & 
                                   (df['scene_name'] == scene) & 
                                   (df['scheme'] == scheme)]
                        if len(subset) > 0:
                            row = {
                                '模型': model,
                                '场景': scene,
                                '方案': scheme,
                                '总样本数': len(subset),
                                '输入Token总量': subset['input_tokens'].sum(),
                                '输出Token总量': subset['output_tokens'].sum(),
                                '总耗时(s)': round(subset['elapsed_time'].sum(), 1)
                            }
                            for dim in dimensions:
                                row[f"{dim['name']}_均值"] = round(subset[dim['name']].mean(), 1)
                                row[f"{dim['name']}_最高"] = round(subset[dim['name']].max(), 1)
                                row[f"{dim['name']}_最低"] = round(subset[dim['name']].min(), 1)
                            overall_data.append(row)
            
            overall_df = pd.DataFrame(overall_data)
            overall_df.to_excel(writer, sheet_name='综合统计', index=False)
        
        self.logger.info(f"最终报告已保存: {report_path}")


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()
