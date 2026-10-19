# 任务六：信息抽取与 JSON 结构化

> 本题为纯文本任务，所有信息均在题干中，无需联网或调用工具。

阅读下面的项目周报，并从中抽取结构化信息：

> 项目“青禾”周报（10 月 14 日）
>
> 负责人：周敏。目标上线日期：11 月 15 日，比原计划推迟一周。已确认预算 48 万元；如增加第三方地图服务，需再申请 5 万元，尚未批准。
>
> 本周完成：用户登录与权限模块联调；完成 12 个核心页面的视觉走查。
>
> 进行中：支付接口改造，完成约 70%，阻塞原因是对方接口文档缺少退款字段。
>
> 尚未开始：数据迁移演练。
>
> 下周计划：10 月 18 日前补齐退款字段联调；10 月 22 日组织第一次全量演练；10 月 25 日提交灰度发布方案。
>
> 需要决策：是否采购第三方地图服务；若采购，由谁承担新增的 5 万元预算。
>
> 风险：测试环境数据库版本与生产不一致，可能导致演练结果失真。
>
> 客户对接人：王磊。

请严格按照以下 JSON 结构输出：

```json
{
  "project_name": "string",
  "owner": "string",
  "target_launch_date": "string",
  "budget_cny_wan": {
    "confirmed": 0,
    "pending": 0
  },
  "completed": ["string"],
  "in_progress": [
    {
      "item": "string",
      "progress": "string",
      "blocker": "string"
    }
  ],
  "not_started": ["string"],
  "next_actions": [
    {
      "action": "string",
      "due": "string"
    }
  ],
  "decisions_needed": ["string"],
  "risks": ["string"],
  "client_contact": "string",
  "unconfirmed_items": ["string"]
}
```

**要求：**

- 严格使用上述字段名和类型；没有对应信息的字段用 `null` 或空数组，不得编造。
- 日期保留原文表述，不补全年份，不擅自换算格式。
- 预算金额用 JSON 数字，单位已由字段名体现；不要把“48 万元”写成字符串。
- “约 70%”这样的不确定表述要保留其不确定性，不要改写成精确值。
- `unconfirmed_items` 只列出原文明确说明“尚未批准/未确认”的事项。
- 只输出一个合法 JSON 对象，不要 Markdown 代码围栏，不要任何解释文字。
