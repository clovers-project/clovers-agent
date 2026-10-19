---
name: weather-report
description: 查询天气预报信息
parameters:
  type: object
  properties:
    city:
      type: string
      description: 需要查询的城市
  required:
    - city
---

## {city}城市天气信息

{weather}

## 分时预报

{forecast}
