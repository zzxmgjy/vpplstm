## 自动调优脚本

`auto_tune_power_model.py` 脚本提供三个主要功能：

1. **超参数调优**：自动搜索最佳超参数
2. **应用最佳参数**：使用已知的良好参数训练模型
3. **微调**：通过集中训练进一步改进现有模型

## 使用方法

### 1. 超参数调优

要为您的模型找到最佳超参数：

```bash
python auto_tune_power_model.py --mode tune --station_id 1716387625733984256 --data_file merged_station_test.csv --n_trials 10 --random_search
```

参数：
- `--mode tune`：在超参数调优模式下运行
- `--station_id`：要训练的站点ID
- `--data_file`：输入数据文件的路径
- `--n_trials`：尝试的超参数组合数量
- `--random_search`：使用随机搜索而非网格搜索（推荐用于更快获得结果）

脚本将：
1. 尝试不同的超参数组合
2. 跟踪表现最佳的参数
3. 将结果保存到 `hyperparameter_tuning_results/`
4. 使用最佳参数训练最终模型

### 2. 应用最佳参数

直接应用已知的良好参数：

```bash
python auto_tune_power_model.py --mode apply --station_id 1716387625733984256 --data_file merged_station_test.csv
```

参数：
- `--mode apply`：应用已知的最佳参数
- `--station_id`：要训练的站点ID
- `--data_file`：输入数据文件的路径

这将使用预定义的优化参数训练模型。

### 3. 微调

微调现有模型：

```bash
python auto_tune_power_model.py --mode finetune --model_dir models/station_1716387625733984256 --station_id 1716387625733984256 --data_file merged_station_test.csv --focus_days 1 2 3
```

参数：
- `--mode finetune`：在微调模式下运行
- `--model_dir`：包含预训练模型的目录
- `--station_id`：要微调的站点ID
- `--data_file`：输入数据文件的路径
- `--epochs`：微调的轮数（默认：50）
- `--lr`：微调的学习率（默认：1e-5）
- `--focus_days`：要重点关注的天数（1-7），例如 `--focus_days 1 2 3` 重点关注第1、2和3天

这将微调模型，特别关注改进指定的天数。

## 示例工作流程

1. 首先，运行超参数调优以找到良好的参数：
   ```bash
   python auto_tune_power_model.py --mode tune --station_id 1716387625733984256 --data_file merged_station_test.csv --n_trials 10 --random_search
   ```

2. 然后，微调生成的模型，重点关注MAPE较高的天数：
   ```bash
   python auto_tune_power_model.py --mode finetune --model_dir models/station_1716387625733984256 --focus_days 1 2 3 --lr 1e-5 --epochs 50
   ```

3. 评估最终模型的性能。

## 获得更好结果的提示

1. 从合理数量的试验（10-20次）开始，找到良好的初始参数
2. 关注影响最大的超参数：
   - `hidden_dim`：隐藏层的大小（更大可以捕获更复杂的模式）
   - `num_layers`：LSTM层的数量（更多层可以建模更复杂的关系）
   - `lr`：学习率（对收敛至关重要）
   - `power_weight` 和 `not_use_power_weight`：两个预测目标之间的平衡

3. 对于微调，关注MAPE值较高的天数（在本例中，第1-5天的MAPE略高）

4. 考虑以下参数范围以获得更好的结果：
   - `hidden_dim`：256-512
   - `num_layers`：2-3
   - `drop_rate`：0.2-0.4
   - `batch_size`：64-128
   - `lr`：1e-4 到 5e-4
   - `power_weight`：1.0-1.5
   - `not_use_power_weight`：0.7-1.0

5. 对于具有强烈季节性模式的时间序列，启用STL分解（`use_stl=True`）

## 解读结果

调优后，检查每天的MAPE结果。如果某些天持续显示较高的误差：

1. 使用微调模式，重点关注这些特定的天数
2. 考虑这些天是否具有特殊特征（周末、假日）可能需要特殊处理
3. 检查这些天的数据，以识别潜在的异常或模式

## 故障排除

- 如果遇到内存问题，减小 `batch_size` 和 `hidden_dim`
- 如果训练太慢，减少 `n_trials` 并使用 `--random_search`
- 如果模型过拟合（验证损失增加而训练损失减少），增加 `drop_rate`
