"""
分层式量化交易系统 v3
====================

架构:
- layers/data:         数据获取与存储
- layers/features:      特征工程管线
- layers/signals:       Alpha 信号层 (8种模型 + Ensemble)
- layers/portfolio:     组合构建层
- layers/risk:          风控与约束层
- layers/rl:            RL 执行优化层
- layers/env:           Gymnasium 交易环境
- layers/backtest:      事件驱动回测引擎
- layers/evaluation:    评估指标与绩效归因
- layers/experiments:   实验管理与超参搜索
- layers/ensemble:      信号融合 (weighted/stacking/voting/regime)
- layers/inference:     在线推理与监控

数据流:
    data → features → signals → ensemble → portfolio → risk → rl → backtest → evaluation
"""

__version__ = "3.0.0"
