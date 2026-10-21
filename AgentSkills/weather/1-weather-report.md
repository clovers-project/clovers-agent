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

时间: {now}
天气: {weather}
温度: {temperature}
湿度: {humidity}
体感温度: {feels_like}
风向: {wind}

## 分时预报

{forecast}
