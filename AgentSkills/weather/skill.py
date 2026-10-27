from clovers_agent import cast_invoker, CloversAgent, Turn, Event
from datetime import datetime, timedelta


@cast_invoker
async def weather_report(agent: CloversAgent, turn: Turn, event: Event, content: str, city: str):
    now = datetime.now()
    kwargs = {}
    kwargs["city"] = city
    resp = await agent.async_client.get(f"https://wttr.in/{city}?format=j1")
    data = resp.json()
    current = data["current_condition"][0]
    kwargs["now"] = now.strftime("%Y-%m-%d %H:%M")
    kwargs["weather"] = current["weatherDesc"][0]["value"]
    kwargs["temperature"] = f"{current['temp_C']}°C"
    kwargs["humidity"] = f"{current['humidity']}%"
    kwargs["feels_like"] = f"{current['FeelsLikeC']}°C"
    kwargs["wind"] = f"{current['winddir16Point']}（{current['winddirDegree']}°）{current['windspeedKmph']} km/h"
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
    return content.format(**kwargs)
