from clovers_agent import CloversAgent, Event
from datetime import datetime, timedelta


async def weather_report(agent: CloversAgent, event: Event, content: str, city: str):
    now = datetime.now()
    kwargs = {}
    kwargs["city"] = city
    resp = await agent.async_client.get(f"https://wttr.in/{city}?format=j1")
    data = resp.json()
    current_condition = data["current_condition"][0]
    kwargs["weather"] = f"""\
时间: {now.strftime("%Y-%m-%d %H:%M")}
天气: {current_condition["weatherDesc"][0]["value"]}
温度: {current_condition["temp_C"]}°C
湿度: {current_condition["humidity"]}%
体感温度: {current_condition["FeelsLikeC"]}°C
风速: {current_condition["winddir16Point"]}（{current_condition["winddirDegree"]}°）{current_condition["windspeedKmph"]} km/h"""
    forecast = ["时间|天气|温度|降水", "-|-|-|-"]
    for day in data.get("weather", []):
        date = datetime.strptime(day["date"], "%Y-%m-%d")
        for hour in day.get("hourly", []):
            minutes = int(hour["time"])
            dt = date + timedelta(minutes=minutes)
            if dt < now:
                continue
            forecast.append(f"{dt.strftime('%Y-%m-%d %H:%M')}|{hour['weatherDesc'][0]['value']}|{hour['tempC']}°C|{hour['precipMM']}mm")
    kwargs["forecast"] = "\n".join(forecast)
    content = content.format(**kwargs)
    await event.send("text", content)
    return content
