# fpga-test-generation
基于小参数LLM的航天FPGA测试生成研究 - 实验数据与复现脚本

## 实验概述

本研究进行了两轮实验：

1. **探索性实验**: 210次（3场景 × 3方案 × 2模型 × 5温度 × 2重复）
2. **正交验证实验**: 840次（4组 × 多配置组合 × 30重复）

## 目录结构

```
github_upload/
├── README.md                          # 本文件
├── experiment.py                      # 探索性实验执行脚本
├── experiment_orthogonal.py           # 正交验证实验执行脚本
├── config.yaml                        # 探索性实验配置文件
├── config_orthogonal.yaml             # 正交验证实验配置文件
├── exploratory_experiment/            # 探索性实验（210次）
│   ├── config.yaml                    # 实验配置
│   ├── raw_outputs/                   # 原始实验结果（JSON格式）
│   │   ├── sceneA_方案1_纯提示词_P1_Qwen3_14B_1.json
│   │   ├── sceneA_方案1_纯提示词_P1_Qwen3_14B_2.json
│   │   └── ... (210个文件)
│   ├── summary_statistics.csv         # 汇总统计
│   └── 实验结果汇总.csv                # 中文汇总
├── orthogonal_experiment/             # 正交验证实验（840次）
│   ├── raw_outputs/                   # 原始实验结果（JSON格式）
│   │   ├── group1_sceneA_方案2-结构化上下文_T0.0_Qwen_Qwen3-14B_1.json
│   │   ├── group4_sceneA_方案1-纯提示词_T0.0_Qwen_Qwen3-14B_1.json
│   │   └── ... (840个文件)
│   ├── results.csv                    # 结果汇总表
│   ├── statistical_report.txt         # 统计分析报告
│   └── summary_by_group.csv           # 按组汇总
└── reproduction_scripts/              # 数据复现脚本
    ├── recalculate_all_paper_data.py
    ├── verify_orthogonal_data.py
    ├── extract_group4_all_combinations.py
    └── statistical_analysis_with_tests.py
```

## 实验配置

### 探索性实验配置 (config.yaml)

```yaml
# 实验场景
scenes:
  - sceneA: 场景A-测控指令解码
  - sceneB: 场景B-数据采集接口
  - sceneC: 场景C-通信协议处理

# 提示词方案
schemes:
  - 方案1: 纯提示词（简洁型）
  - 方案2: 结构化上下文
  - 方案3: 增强型上下文

# 模型配置
models:
  - Qwen3-14B
  - Qwen3-32B

# 温度参数
temperatures: [0.0, 0.3, 0.6, 0.9, 1.2]

# 评价维度及权重
weights:
  格式合规率: 0.15
  条目对应率: 0.10
  需求覆盖率: 0.15
  测试项有效性: 0.25
  输出描述质量: 0.20
  无幻觉率: 0.15
```

### 正交验证实验配置 (config\_orthogonal.yaml)

```yaml
# 4组正交实验
groups:
  - group1: 温度效应分析
  - group2: 方案效应分析
  - group3: 模型对比分析
  - group4: 交互效应分析

# 每组30次重复
repeats: 30
```

## 实验结果文件格式

每个JSON文件包含以下字段：

```json
{
  "group": "group4",
  "scene_id": "sceneA",
  "scene_name": "场景A-测控指令解码",
  "scheme": "方案1-纯提示词",
  "model": "Qwen/Qwen3-14B",
  "temperature": 0.0,
  "top_p": 1.0,
  "repeat_idx": 1,
  "input_tokens": 322,
  "output_tokens": 1934,
  "elapsed_time": 59.57,
  "output_content": "...生成的测试项内容...",
  "success": true,
  "error": null,
  "格式合规率": 100,
  "条目对应率": 100,
  "需求覆盖率": 95,
  "测试项有效性": 90,
  "输出描述质量": 92,
  "无幻觉率": 100
}
```

## 关键实验结果

### 表7: 交互效应分析（Group4）

| 组合 | 模型  | 方案  | 温度  | 综合得分           |
| -- | --- | --- | --- | -------------- |
| 1  | 14B | 方案1 | 0.0 | **94.82±2.25** |
| 2  | 14B | 方案2 | 0.6 | **92.49±3.11** |
| 3  | 14B | 方案3 | 1.2 | **90.34±3.29** |
| 4  | 32B | 方案1 | 0.6 | **94.22±3.36** |
| 5  | 32B | 方案2 | 1.2 | **7.81±15.69** |
| 6  | 32B | 方案3 | 0.0 | **87.42±4.58** |

### 表8: 人工编制与LLM最佳配置对比

| 评价维度     | 人工编制     | LLM最佳配置（14B+方案1+T=0.0） | 差距       |
| -------- | -------- | ---------------------- | -------- |
| 格式合规率    | 100.0    | 93.50±7.56             | -6.5     |
| 条目对应率    | 100.0    | 100.00±0.00            | +0.0     |
| 需求覆盖率    | 96.5     | 96.67±3.30             | +0.2     |
| 测试项有效性   | 93.0     | 92.07±3.19             | -0.9     |
| 输出描述质量   | 92.5     | 92.17±2.32             | -0.3     |
| 无幻觉率     | 100.0    | 99.00±2.75             | -1.0     |
| **综合得分** | **97.0** | **94.82±2.25**         | **-2.2** |

## 复现步骤

### 环境要求

```bash
# Python 3.8+
pip install pandas numpy scipy matplotlib pyyaml requests
```

### 数据验证

```bash
# 验证正交实验数据
python reproduction_scripts/verify_orthogonal_data.py

# 提取Group4所有组合数据
python reproduction_scripts/extract_group4_all_combinations.py

# 重新计算论文所有表格数据
python reproduction_scripts/recalculate_all_paper_data.py

# 统计检验分析
python reproduction_scripts/statistical_analysis_with_tests.py
```

### 运行实验（如需重新执行）

```bash
# 设置API密钥
export SILICONFLOW_API_KEY="your_api_key"

# 运行探索性实验
python experiment.py --config config.yaml

# 运行正交验证实验
python experiment_orthogonal.py --config config_orthogonal.yaml
```

## API参数说明

- **API端点**: <https://api.siliconflow.cn/v1/chat/completions>
- **模型**: Qwen/Qwen3-14B, Qwen/Qwen3-32B
- **温度参数**: 0.0, 0.3, 0.6, 0.9, 1.2
- **top\_p**: 1.0
- **随机种子**: 未固定（使用模型默认随机性）

## 评价指标说明

| 指标     | 说明             | 权重  |
| ------ | -------------- | --- |
| 格式合规率  | 输出JSON格式是否符合规范 | 15% |
| 条目对应率  | 测试项是否与需求条目对应   | 10% |
| 需求覆盖率  | 是否覆盖所有需求点      | 15% |
| 测试项有效性 | 测试项是否正确有效      | 25% |
| 输出描述质量 | 测试描述是否清晰完整     | 20% |
| 无幻觉率   | 是否包含虚构内容       | 15% |

## 引用信息

如果您使用了本数据集，请引用：

```bibtex
@article{fpga_test_generation_2025,
  title={基于小参数LLM的航天器FPGA测试项自动化生成研究},
  author={作者姓名},
  journal={期刊名称},
  year={2025},
  doi={待生成}
}
```

## 联系信息

如有问题或建议，请通过GitHub Issues联系。

项目数据量过大，仅上传部分进行示例，如需要完整数据，请联系cocoa18\@vip.qq.com

## 许可证

本数据集采用 CC BY 4.0 许可证。
