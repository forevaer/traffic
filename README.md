# 数据整理
- [create_data_file.py](prepare/create_data_file.py)

标准命名，同时生成数据目录`csv`
# 数据转换
- `torch`格式对接： [dataset](entity/dataset.py)
- `前置数据处理`: [transform](config/trans.py)
# 阶段操作
- [train](ops/train.py)
- [test](ops/test.py)
- [predict](ops/predict.py)
# 常用配置
- [config](config/config.py)
# 入口
- [entrance](entrance/entrance.py)