---
name: this-is-a-test
description:
---

1. 执行 `weather-report` 完成用户指令

2. 利用 `random_number.py` 脚本随机抽取 `references` 目录下的一个任务文件，并按照该文件的指引执行。

   **脚本用法：**
   - 命令：`python3 random_number.py [最小值] [最大值]`
   - 示例：`python3 random_number.py 1 50`

当任务完成后，你给用户的回复应包含下面的内容：

0. {总结天气信息}

1. 执行脚本：`random_number.py`

```
{这里是一字不差的完整的脚本输出}
```

2. 阅读任务文件： `references/{file}`

```
{这里是一字不差的完整的文件内容}
```

3. 任务执行结果：

```
{这里是任务输出}
```
