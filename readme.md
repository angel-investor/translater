# SeqSeq模型英译法

## 1 任务目的：

```properties
目的: 给定一段英文，翻译为法文
典型的文本生成任务: 每个时间步去预测应该属于哪个法文单词
```

## 2 数据格式

- 注意：两列数据，第一列是英文文本，第二列是法文文本，中间用制表符号"\t"隔开

```properties
i am from brazil .  je viens du bresil .
i am from france .  je viens de france .
i am from russia .  je viens de russie .
i am frying fish .  je fais frire du poisson .
i am not kidding .  je ne blague pas .
```

## 3 任务实现流程

```properties
1. 获取数据:案例中是直接给定的
2. 数据预处理: 脏数据清洗、数据格式转换、数据源Dataset的构造、数据迭代器Dataloader的构造
3. 模型搭建: 编码器和解码器等一系列模型
4. 模型训练
5. 模型评估（测试）
6. 注意力可视化
7. 模型上线---API接口
```