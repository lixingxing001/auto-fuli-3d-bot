# 福彩3D量化分析机器人

Lee，这个项目当前采用 Python 第一版。目标是先建立可验证的量化链路，再讨论策略复杂度。

## 当前定位

* 用途: 历史开奖管理、特征统计、候选号码评分、滚动回测
* 边界: 不接真实购彩，不做自动下单，不承诺必中
* 第一原则: 所有策略先过回测，无法回测的想法先降级为假设

## 数据格式

历史文件使用 CSV，字段固定为:

```csv
issue,date,number
2026001,2026-01-01,314
```

默认路径是 `data/history.csv`。仓库里提供了 `data/history.sample.csv`，它只是演示格式。真实分析前要替换成可靠历史开奖数据。

## 快速运行

```powershell
Copy-Item data/history.sample.csv data/history.csv
python main.py summary
python main.py recommend --top 20
python main.py backtest --training-window 30 --top 20
python main.py explain 314
```

拉取中国福彩网公开接口数据:

```powershell
python main.py fetch --output data/history.csv
python main.py summary
python main.py backtest --training-window 500 --top 20
python main.py experiment --limit-rows 1200
python main.py benchmark --limit-rows 1200 --top 20 --training-window 300 --recent-window 60
python main.py benchmark --limit-rows 1200 --top 20 --training-window 300 --recent-window 60 --monthly --monthly-year 2025 --output-dir reports/benchmark_2025_monthly
python main.py attribution --year 2025 --months 3,5 --compare-years 2024,2026 --output-dir reports/attribution_2025_03_05
python main.py rulebacktest --rules "ones=5;sum=15;pattern=zuliu;ones=5,sum=15;ones=5,pattern=zuliu;sum=15,pattern=zuliu" --years 2024,2025,2026 --output-dir reports/rulebacktest
python main.py rulerecency --rules "ones=5,sum=15,pattern=zuliu;sum=15,pattern=zuliu;pattern=zuliu" --windows 60,120,240,360 --output-dir reports/rulerecency
python main.py gate --rules "ones=5,sum=15,pattern=zuliu;sum=15,pattern=zuliu;pattern=zuliu" --windows 60,120,240,360 --gate-windows 60,120 --output-dir reports/gate
python main.py calibration --limit-rows 1200 --training-window 300 --recent-window 60 --output-dir reports/calibration
python main.py ablation --limit-rows 1200 --training-window 300 --recent-window 60 --output-dir reports/ablation
python main.py variantstress --variants baseline,no_repeat_penalty --limit-rows 1200 --training-window 300 --recent-window 60 --output-dir reports/variantstress
python main.py delivery --output-dir reports/delivery
python main.py targetcoverage --target-rate 0.65 --limit-rows 1200 --training-window 300 --recent-window 60 --output-dir reports/targetcoverage
python main.py daily --output-dir reports/daily
python main.py review --predictions-dir reports/daily/snapshots --output-dir reports/review
```

完整历史实验会明显更慢:

```powershell
python main.py experiment --limit-rows 0
```

也可以安装成命令:

```powershell
python -m pip install -e .
fuli3d --data data/history.csv recommend --top 20
```

## 已实现模块

* `src/fuli3d_bot/features.py`: 号码标准化、和值、跨度、形态、奇偶、大小
* `src/fuli3d_bot/fetcher.py`: 中国福彩网公开接口拉取和 CSV 落盘
* `src/fuli3d_bot/validation.py`: 重复期号、期号跳跃、日期跳跃、排序校验
* `src/fuli3d_bot/stats.py`: 长期和近期统计、遗漏统计、近期重号统计
* `src/fuli3d_bot/strategy.py`: 000 到 999 全量评分和候选排序
* `src/fuli3d_bot/backtest.py`: walk forward 滚动回测、随机基准、命中率、ROI、最大回撤、最大连亏
* `src/fuli3d_bot/experiment.py`: 批量参数实验、排行榜、JSON 和 Markdown 报告
* `src/fuli3d_bot/baselines.py`: 热号、冷号、和值、跨度、形态、固定随机等弱基准
* `src/fuli3d_bot/benchmark.py`: 综合策略和弱基准对照、年度与月度分段报告
* `src/fuli3d_bot/attribution.py`: 命中号码归因和对照年份特征压力检查
* `src/fuli3d_bot/rules.py`: 归因特征过滤规则回测和时效性检测
* `src/fuli3d_bot/gate.py`: 规则闸门，把 active、watch、blocked 分开，短窗口未过关的规则不参与当前过滤
* `src/fuli3d_bot/calibration.py`: 基础评分校准，检查真实开奖号在历史滚动评分中的排名分布
* `src/fuli3d_bot/ablation.py`: 权重消融，检查评分项的贡献和拖累
* `src/fuli3d_bot/variantstress.py`: 权重变体压力测试，按年度和近期窗口复核候选变体
* `src/fuli3d_bot/delivery.py`: 交付状态汇总，明确当前是否允许进入推荐验收
* `src/fuli3d_bot/targetcoverage.py`: 目标覆盖率测算，说明达到指定命中率需要覆盖多少号码以及期望亏损
* `src/fuli3d_bot/daily.py`: 每日预测页面，展示主预测号码、备选号、不同玩法奖金和回测命中率
* `src/fuli3d_bot/review.py`: 每日预测快照复盘，统计直选、组选、Top3 和 Top5 命中
* `src/fuli3d_bot/cli.py`: 所有命令行入口

## 严格边界

这个项目的评分逻辑只说明某个号码为什么在当前规则下排得更高。它不能证明下一期概率发生变化。若回测不能长期覆盖不同时间段，任何高分都只能当作噪声候选。

## 校准判读

`calibration` 会记录每一期真实开奖号在滚动评分中的排名，再检查 TopN 命中、近期窗口和评分分桶。若 mean_actual_rank 没有低于 500，或 Top20 的 z 值不能跨窗口为正，基础评分只能保留为观察工具。若分桶 lift 没有从高分桶到低分桶大体下降，说明排序结构仍然不稳。

## 当前交付边界

`delivery` 是当前最终状态入口。若输出 `mode=analysis_only` 或 `can_recommend=false`，项目只交付为分析观察工具。只有规则闸门、基础评分校准和变体压力测试同时通过，才允许进入推荐验收。

## 65% 目标说明

`targetcoverage --target-rate 0.65` 会把目标换算为 Top650 覆盖。这个口径能提高命中率，同时会覆盖 1000 个直选号码里的 650 个，期望 ROI 仍为 -48%。这属于覆盖率方案，没有证明模型预测能力提高。

下一步优先级:

1. 先看 `reports/benchmark/benchmark_results.md` 的年度分段
2. 如果综合策略只靠单一年份拉升，先调整策略再谈推荐
3. 如果综合策略不能稳定赢过弱基准，停止增加复杂模型
4. 如果归因规则在最近 60 和 120 期退潮，先进入 blocked 或 watch
5. 只有跨年度、弱基准和短窗口闸门都过关，才输出每日推荐报告
6. API 服务和消息机器人入口放在模型验证之后
