# -*- coding: utf-8 -*-
"""
核心模块包 - 伴随患者分析

本包提供伴随患者分析系统的所有核心功能模块，实现从时空接触检测到患者角色识别的完整工作流。

模块架构：
===========
1. geo_utils（地理计算工具层）
   - 提供基础的地理距离计算和空间索引功能
   - 基于Haversine公式和BallTree空间索引
   - 支持高效的近邻查询和时空过滤

2. contact_detector（接触检测层）
   - 实现多模式接触检测（医院、居住地、全程）
   - 基于geo_utils的空间索引进行时空接触分析
   - 支持多进程并行处理和时间窗口优化

3. pair_matcher（对匹配层）
   - 执行多数据源患者对的交叉匹配
   - 识别同时满足多个条件的患者对
   - 支持自动文件扫描和批量处理

4. record_extractor（记录提取层）
   - 从原始轨迹数据中提取目标患者记录

5. patient_classifier（患者分类层）
   - 基于网络图论分析患者接触网络
   - 自动识别网络结构类型和患者角色
   - 计算伴随比例并生成分类结果

数据流向：
==========
原始轨迹数据
    ↓
[contact_detector] 时空接触检测
    ↓
患者对数据
    ↓
[pair_matcher] 多条件匹配
    ↓
匹配患者对
    ↓
[record_extractor] 轨迹提取
    ↓
完整轨迹数据
    ↓
[contact_detector] 全程接触检测
    ↓
最终患者对
    ↓
[patient_classifier] 网络分析与角色分类
    ↓
患者/伴随患者分类结果

导出接口：
==========
工具函数：
  - haversine_distance: 计算两点球面距离
  - to_radians: 米转弧度转换
  - build_ball_tree: 构建空间索引
  - find_nearby_points: 范围查询

配置类：
  - DetectorConfig: 接触检测器配置

核心类：
  - ContactDetector: 接触检测器
  - PairMatcher: 患者对匹配器
  - RecordExtractor: 记录提取器
  - RatioCalculator: 比例计算器

使用示例：
==========
>>> # 导入核心模块
>>> from core import ContactDetector, DetectorConfig
>>>
>>> # 配置并运行接触检测
>>> config = DetectorConfig.for_hospital(
...     input_dir='data/hospital_trajectories',
...     output_dir='data/hospital_companions',
...     distance_threshold=50,
...     time_threshold_minutes=30
... )
>>> detector = ContactDetector(config)
>>> results = detector.detect_from_file('Beijing_2024-01-01.csv')
>>>
>>> # 或使用完整流水线（参见pipeline.py）
>>> from pipeline import CompanionAnalysisPipeline, PipelineConfig
>>> pipeline = CompanionAnalysisPipeline(PipelineConfig())
>>> pipeline.run_full_pipeline()

依赖关系：
==========
- geo_utils: 无内部依赖（基础层）
- contact_detector: 依赖 geo_utils
- pair_matcher: 无内部依赖（独立模块）
- record_extractor: 无内部依赖（独立模块）
- patient_classifier: 无内部依赖（独立模块）

外部依赖：
==========
- numpy: 数值计算
- polars: 高性能数据处理
- scikit-learn: BallTree空间索引
- networkx: 网络图论分析
- haversine: Haversine距离计算

版本：v1.0.0
"""

from .geo_utils import (
    haversine_distance,
    to_radians,
    build_ball_tree,
    find_nearby_points,
)
from .contact_detector import DetectorConfig, ContactDetector
from .pair_matcher import PairMatcher
from .record_extractor import RecordExtractor
from .patient_classifier import RatioCalculator

__all__ = [
    # 地理工具
    'haversine_distance',
    'to_radians',
    'build_ball_tree',
    'find_nearby_points',
    # 配置类
    'DetectorConfig',
    # 核心类
    'ContactDetector',
    'PairMatcher',
    'RecordExtractor',
    'RatioCalculator',
]
