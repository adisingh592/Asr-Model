"""
Free Live Weather Tool for Voice Assistant using Open-Meteo REST API.
Zero API keys required. Free, fast, and globally accurate.
"""

import json
import urllib.parse
import urllib.request
from typing import Dict, Optional

# WMO Weather interpretation codes (WW)
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_coordinates_for_city(city: str) -> Optional[Dict[str, float]]:
    """Geocodes a city name to latitude and longitude using Open-Meteo Geocoding."""
    encoded_city = urllib.parse.quote(city.strip())
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    
    req = urllib.request.Request(url, headers={"User-Agent": "ESP32-Voice-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results")
            if results and len(results) > 0:
                return {
                    "name": results[0].get("name", city),
                    "latitude": results[0]["latitude"],
                    "longitude": results[0]["longitude"],
                    "country": results[0].get("country", "")
                }
    except Exception as e:
        print(f"[Weather Tool] Geocoding error for '{city}': {e}")
    return None


def get_live_weather(location: str = "Delhi") -> str:
    """
    Fetches real-time weather conditions for the specified location.
    
    Returns:
        A concise, speech-friendly weather summary.
    """
    if not location or location.strip() == "":
        location = "Delhi"

    geo = get_coordinates_for_city(location)
    if not geo:
        return f"Sorry, I could not find location coordinates for {location}."

    lat = geo["latitude"]
    lon = geo["longitude"]
    place_name = geo["name"]

    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        f"&timezone=auto"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "ESP32-Voice-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            current = data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            code = current.get("weather_code", 0)
            wind = current.get("wind_speed_10m")

            condition = WMO_CODES.get(code, "Clear")
            
            return (
                f"The weather in {place_name} is currently {round(temp)}°C and {condition.lower()}, "
                f"with {round(humidity)}% humidity and wind at {round(wind)} kilometers per hour."
            )
    except Exception as e:
        return f"Sorry, I was unable to retrieve live weather data right now due to a network error: {e}"


if __name__ == "__main__":
    print(get_live_weather("Delhi"))
    print(get_live_weather("London"))
    print(get_live_weather("New York"))
