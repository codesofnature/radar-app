import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import concurrent.futures
import math
import io
import time
import logging
import json
import numpy as np
from PIL import Image
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from functools import lru_cache

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Instant Radar", layout="wide", page_icon="⚡", initial_sidebar_state="collapsed")

# --- Streamlit UI Tweaks ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.terrain-blend { mix-blend-mode: multiply; opacity: 0.65; }
.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}
.stApp {
    background-color: #cce4f0 !important;
}
iframe {
    border: none;
}
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
LOCAL_TZ = ZoneInfo("America/New_York")
BBOX = "-14000000,2630000,-7400000,6480000"
WIDTH = 1200
HEIGHT = 700
RADAR_RES_FACTOR = 1.5
RADAR_W = int(WIDTH * RADAR_RES_FACTOR)
RADAR_H = int(HEIGHT * RADAR_RES_FACTOR)
MINUTES_OFFSETS = list(range(0, 18 * 60, 15)) + list(range(18 * 60, 49 * 60, 60))

# --- NASA-Quality Embedded Images (Direct URLs) ---
@lru_cache(maxsize=None)
def get_embedded_sun_image():
    """Returns ultra-realistic NASA sun image as data URI"""
    try:
        sun_urls = [
            "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg",
        ]
        for url in sun_urls:
            try:
                response = requests.get(url, timeout=8)
                if response.status_code == 200:
                    b64 = base64.b64encode(response.content).decode('utf-8')
                    return f"data:image/jpeg;base64,{b64}"
            except:
                continue
    except Exception as e:
        logger.warning(f"Failed to fetch sun image: {e}")
    return ""

# Embedded moon image (base64 JPEG) baked directly into the script
@lru_cache(maxsize=None)
def get_embedded_moon_image():
    """Returns high-resolution NASA moon image as data URI (reliable URL)"""
    # Direct NASA/USGS-style full disk moon image (high res, orthographic view)
    moon_url = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg"
    try:
        response = requests.get(moon_url, timeout=10)
        if response.status_code == 200:
            b64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to fetch moon image: {e}")
    
    # Fallback: smaller public moon image
    return "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/1024px-FullMoon2010.jpg"

def mercator_to_latlon(x, y):
    r = 6378137.0
    lon = math.degrees(x / r)
    lat = math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2.0)
    return lat, lon

xmin, ymin, xmax, ymax = map(float, BBOX.split(","))
lat_min, lon_min = mercator_to_latlon(xmin, ymin)
lat_max, lon_max = mercator_to_latlon(xmax, ymax)
MAP_BOUNDS = f"[[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]]"

ROWS, COLS = 14, 20
lats = np.linspace(lat_max, lat_min, ROWS)
lons = np.linspace(lon_min, lon_max, COLS)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_openmeteo_data():
    try:
        lat_list = []
        lon_list = []
        for lat in lats:
            for lon in lons:
                lat_list.append(f"{lat:.2f}")
                lon_list.append(f"{lon:.2f}")
        
        # Added relativehumidity_2m to the hourly request
        url = f"https://api.open-meteo.com/v1/forecast?latitude={','.join(lat_list)}&longitude={','.join(lon_list)}&hourly=temperature_2m,relativehumidity_2m&temperature_unit=fahrenheit&past_days=1&forecast_days=3"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Open-Meteo error: {resp.text}")
        return []
    except Exception as e:
        logger.error(f"Open-Meteo request failed: {e}")
        return []

def generate_temp_overlay(target_dt, om_data):
    if not om_data:
        return "", []
    try:
        target_hour = target_dt.replace(minute=0, second=0, microsecond=0)
        if target_dt.minute >= 30:
            target_hour += timedelta(hours=1)
        target_time_str = target_hour.strftime("%Y-%m-%dT%H:00")

        time_arr = om_data[0].get("hourly", {}).get("time", [])
        try:
            t_idx = time_arr.index(target_time_str)
        except ValueError:
            if time_arr:
                target_ts = target_hour.timestamp()
                best_idx = 0
                min_diff = float('inf')
                for i, t in enumerate(time_arr):
                    try:
                        dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
                        diff = abs(dt.timestamp() - target_ts)
                        if diff < min_diff:
                            min_diff = diff
                            best_idx = i
                    except:
                        pass
                t_idx = best_idx
            else:
                t_idx = -1

        temps = []
        grid_data = []
        idx = 0
        for r, lat in enumerate(lats):
            for c, lon in enumerate(lons):
                val = 50.0
                hum = 50.0
                if t_idx != -1 and idx < len(om_data):
                    loc_temps = om_data[idx].get("hourly", {}).get("temperature_2m", [])
                    loc_hums = om_data[idx].get("hourly", {}).get("relativehumidity_2m", [])
                    
                    if t_idx < len(loc_temps) and loc_temps[t_idx] is not None:
                        val = loc_temps[t_idx]
                    if t_idx < len(loc_hums) and loc_hums[t_idx] is not None:
                        hum = loc_hums[t_idx]
                        
                temps.append(val)
                # Added humidity to the grid data payload
                grid_data.append({"lat": lat, "lon": lon, "val": round(val), "hum": round(hum)})
                idx += 1

        temp_grid = np.array(temps, dtype=np.float32).reshape((ROWS, COLS))
        grid_img = Image.fromarray(temp_grid, mode="F")
        large_grid_img = grid_img.resize((RADAR_W, RADAR_H), Image.BICUBIC)
        smooth_temps = np.array(large_grid_img)

        norm = np.clip((smooth_temps - 20) / 100.0, 0, 1)
        norm = np.floor(norm * 10) / 10.0

        colors = np.zeros((RADAR_H, RADAR_W, 4), dtype=np.uint8)
        m1 = norm < 0.25
        m2 = (norm >= 0.25) & (norm < 0.50)
        m3 = (norm >= 0.50) & (norm < 0.75)
        m4 = norm >= 0.75

        f1 = norm / 0.25
        f2 = (norm - 0.25) / 0.25
        f3 = (norm - 0.50) / 0.25
        f4 = (norm - 0.75) / 0.25

        colors[m1, 2] = 139 + 116 * f1[m1]
        colors[m2, 1] = 255 * f2[m2]
        colors[m2, 2] = 255 - 255 * f2[m2]
        colors[m3, 0] = 255 * f3[m3]
        colors[m3, 1] = 255 - 115 * f3[m3]
        colors[m4, 0] = 255 - 116 * f4[m4]
        colors[m4, 1] = 140 - 140 * f4[m4]
        colors[:, :, 3] = 45

        edge = np.zeros(norm.shape, dtype=bool)
        edge[1:, :] |= norm[1:, :] != norm[:-1, :]
        edge[:-1, :] |= norm[1:, :] != norm[:-1, :]
        edge[:, 1:] |= norm[:, 1:] != norm[:, :-1]
        edge[:, :-1] |= norm[:, 1:] != norm[:, :-1]
        colors[edge] = [0, 0, 0, 40]

        final_img = Image.fromarray(colors, mode="RGBA")
        buf = io.BytesIO()
        final_img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}", grid_data
    except Exception as e:
        logger.error(f"Open-Meteo rendering error: {e}")
        return "", []

def process_radar_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        data = np.array(img)
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        out = np.zeros_like(data)
        valid = a >= 100
        is_severe = valid & (r > 200) & (g < 80)
        is_mod = valid & (r > 200) & (g >= 80) & (g < 220) & (b < 100)
        is_light = valid & ~is_severe & ~is_mod

        out[is_severe, 0], out[is_severe, 1], out[is_severe, 2] = 255, 0, 0
        out[is_mod, 0], out[is_mod, 1], out[is_mod, 2] = 0, 0, 255
        out[is_light, 0], out[is_light, 1], out[is_light, 2] = 0, 255, 255

        new_a = np.where(a < 200, a + 55, 255)
        out[valid, 3] = new_a[valid]

        final_img = Image.fromarray(out, 'RGBA')
        buf = io.BytesIO()
        final_img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return img_bytes

@st.cache_data(ttl=300)
def get_model_init_time():
    try:
        res = requests.get("https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/refd_1080.json", timeout=5)
        dt_str = res.json()["model_init_utc"]
        if not dt_str.endswith("Z"):
            dt_str += "Z"
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception as e:
        now = datetime.now(timezone.utc)
        return now.replace(minute=0, second=0, microsecond=0)

def fetch_frame_data(url_info, om_data, max_retries=2):
    frame_time, time_str, url = url_info
    radar_src = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                processed_bytes = process_radar_image(resp.content)
                b64 = base64.b64encode(processed_bytes).decode("utf-8")
                radar_src = f"data:image/png;base64,{b64}"
                break
            elif resp.status_code == 404:
                return None
        except Exception:
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                
    if radar_src:
        temp_img, temp_grid = generate_temp_overlay(frame_time, om_data)
        return {
            "dt": frame_time,
            "time": time_str,
            "radarImg": radar_src,
            "tempImg": temp_img,
            "tempGrid": temp_grid
        }
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_frame():
    now = datetime.now(timezone.utc)
    om_data = fetch_openmeteo_data()
    live_wms_url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0q-900913&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&SRS=EPSG:3857&BBOX={BBOX}"
    live_label = now.astimezone(LOCAL_TZ).strftime("%a, %b %d - %I:%M %p (Live)")
    frame = fetch_frame_data((now, live_label, live_wms_url), om_data)
    if frame:
        return [frame]
    return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast_frames():
    init_time = get_model_init_time()
    now = datetime.now(timezone.utc)
    om_data = fetch_openmeteo_data()
    frames_data = []

    for attempt in range(3):
        hrrr_frames = []
        urls_to_fetch = []
        for mins_offset in MINUTES_OFFSETS:
            frame_time = init_time + timedelta(minutes=mins_offset)
            if frame_time > now:
                layer_name = f"refd_{str(mins_offset).zfill(4)}"
                url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/hrrr/refd.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer_name}&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&CRS=EPSG:3857&BBOX={BBOX}"
                local_time = frame_time.astimezone(LOCAL_TZ)
                time_str = local_time.strftime("%a, %b %d - %I:%M %p")
                urls_to_fetch.append((frame_time, time_str, url))

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(fetch_frame_data, u, om_data): u for u in urls_to_fetch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    hrrr_frames.append(result)
                    
        if len(hrrr_frames) > 5:
            hrrr_frames.sort(key=lambda x: x["dt"])
            frames_data.extend(hrrr_frames)
            return frames_data
        init_time = init_time - timedelta(hours=1)
        
    return frames_data

def generate_map_html(radar_frames, mode="live"):
    is_forecast = (mode == "forecast")
    
    # Get embedded NASA images
    sun_img_data = get_embedded_sun_image()
    moon_img_data = get_embedded_moon_image()

    frames_js_list = []
    for f in radar_frames:
        grid_json = json.dumps(f['tempGrid'])
        ts_val = int(f["dt"].timestamp() * 1000)
        frames_js_list.append(f"""{{ ts: {ts_val}, time: '{f["time"]}', radarImg: '{f["radarImg"]}', tempImg: '{f["tempImg"]}', tempGrid: {grid_json} }}""")
    
    js_frames_array = ",\n".join(frames_js_list)

    # Sun HTML - NASA quality image
    sun_html = f"""
    <div id="sun-indicator">
        <div class="sun-glow"></div>
        <div class="sun-image-container">
            <img class="sun-image" src="{sun_img_data}" alt="Sun" onerror="this.style.display='none'">
        </div>
    </div>
    """

    # Moon HTML - NASA quality image
    moon_html = f"""
    <div id="moon-indicator">
        <div class="moon-glow"></div>
        <div class="moon-image-container">
            <img class="moon-image" src="{moon_img_data}" alt="Moon" onerror="this.style.display='none'">
            <div class="moon-phase-shadow" id="moonShadow"></div>
        </div>
    </div>
    """

    return f"""
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/suncalc/1.8.0/suncalc.min.js"></script>
<style>
body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, sans-serif; overflow: hidden; }}
#map-container {{ position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; }}
#map {{ width: 100%; height: 100%; background: transparent; }}
.radar-blend {{ mix-blend-mode: multiply; }}
.radar-blend img {{ filter: drop-shadow(-10px 10px 8px rgba(0, 0, 0, 0.5)); }}
.temp-label {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-size: 20px;
    font-weight: 200;
    letter-spacing: 0.2px;
    text-align: center;
    pointer-events: none;
    margin-top: -8px;
    text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.8), 0px 0px 3px rgba(0, 0, 0, 0.5);
}}
#layer-selector {{
    position: absolute; left: 25px; bottom: 25px; z-index: 9999;
    background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px); padding: 12px 18px; border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15); display: flex; flex-direction: column;
    gap: 12px; font-size: 15px; font-weight: 700; color: #0f172a;
}}
.radio-label {{ display: flex; align-items: center; gap: 8px; cursor: pointer; }}
.radio-label input[type="radio"] {{ accent-color: #4f46e5; cursor: pointer; width: 32px; height: 32px; }}
#time-display {{
    position: absolute; top: 15px; left: 50%; transform: translateX(-50%); z-index: 9999;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.55) 0%, rgba(255, 255, 255, 0.15) 100%);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-top: 2px solid rgba(255, 255, 255, 0.9); border-left: 2px solid rgba(255, 255, 255, 0.6);
    border-bottom: 1px solid rgba(255, 255, 255, 0.2); border-right: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25), 0 5px 15px rgba(0, 0, 0, 0.15),
                inset 0 3px 5px rgba(255, 255, 255, 0.9), inset 0 -3px 5px rgba(0, 0, 0, 0.08);
    padding: 10px 24px; border-radius: 16px; font-size: 22px; font-weight: 800;
    color: #0f172a; white-space: nowrap; letter-spacing: -0.5px;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.9); overflow: hidden; 
}}
#time-display::after {{
    content: ""; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0) 100%);
    transform: skewX(-25deg); animation: glassGlare 6s infinite; pointer-events: none;
}}
@keyframes glassGlare {{ 0% {{ left: -100%; }} 15% {{ left: 200%; }} 100% {{ left: 200%; }} }}
#left-controls {{
    position: absolute; left: 15px; top: 50%; transform: translateY(-50%); z-index: 9999;
    background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px); padding: 15px 12px; border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15); display: flex; flex-direction: column;
    align-items: center; gap: 25px;
}}
#playBtn {{
    background: #4f46e5; border: none; color: white; width: 34px; height: 34px;
    border-radius: 50%; cursor: pointer; font-size: 14px; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
    transition: background 0.2s; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}}
#playBtn:hover:not(:disabled) {{ background: #4338ca; }}
#playBtn:disabled {{ background: #94a3b8; cursor: wait; }}
.slider-container {{ position: relative; width: 20px; height: 250px; }}
input[type="range"] {{
    -webkit-appearance: none; background: transparent; cursor: pointer; margin: 0;
    position: absolute; width: 250px; height: 20px; top: 50%; left: 50%;
    transform: translate(-50%, -50%) rotate(-90deg);
}}
input[type="range"]:focus {{ outline: none; }}
input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%;
    background: #4f46e5; margin-top: -6px; box-shadow: 0 1px 4px rgba(0,0,0,0.4); border: 2px solid #fff; cursor: grab;
}}
input[type="range"]:disabled::-webkit-slider-thumb {{ background: #94a3b8; cursor: wait; }}
input[type="range"]::-webkit-slider-runnable-track {{ width: 100%; height: 4px; background: #cbd5e1; border-radius: 2px; }}
#loading-overlay {{ position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255,255,255,0.9);
    display: flex; align-items: center; justify-content: center; z-index: 10000; font-size: 18px; font-weight: bold;
    color: #334155; transition: opacity 0.4s ease-out; }}
#loading-overlay.hidden {{ opacity: 0; pointer-events: none; }}
.leaflet-top.leaflet-right {{ top: 15px; right: 15px; }}
body.forecast-mode {{ opacity: 0; transition: opacity 0.7s ease-in-out; }}
body.forecast-mode.loaded {{ opacity: 1; }}
body.forecast-mode #loading-overlay {{ display: none !important; }}

/* --- NASA-Quality Sun Indicator --- */
#sun-indicator {{ position: absolute; top: 25px; left: 25px; z-index: 9999; width: 120px; height: 120px; transition: opacity 0.5s ease, transform 0.5s ease; pointer-events: none; opacity: 0; }}
.sun-image-container {{ position: relative; width: 100%; height: 100%; border-radius: 50%; overflow: hidden; animation: sunPulse 4s ease-in-out infinite; }}
.sun-image {{ width: 100%; height: 100%; border-radius: 50%; object-fit: cover; transform: scale(1.4); filter: drop-shadow(0 0 30px rgba(255, 200, 0, 0.9)) drop-shadow(0 0 60px rgba(255, 140, 0, 0.6)) drop-shadow(0 0 90px rgba(255, 69, 0, 0.4)); }}
.sun-glow {{ position: absolute; top: -30%; left: -30%; width: 160%; height: 160%; border-radius: 50%; background: radial-gradient(circle, rgba(255, 200, 0, 0.5) 0%, rgba(255, 140, 0, 0.3) 40%, transparent 70%); animation: glowPulse 3s ease-in-out infinite; pointer-events: none; }}
@keyframes sunPulse {{ 0%, 100% {{ transform: scale(1) rotate(0deg); }} 50% {{ transform: scale(1.08) rotate(2deg); }} }}
@keyframes glowPulse {{ 0%, 100% {{ opacity: 0.6; transform: scale(1); }} 50% {{ opacity: 1; transform: scale(1.1); }} }}

/* --- NASA-Quality Moon Indicator --- */
#moon-indicator {{ position: absolute; bottom: 25px; right: 25px; z-index: 9999; width: 100px; height: 100px; transition: opacity 0.5s ease; pointer-events: none; }}
.moon-image-container {{ position: relative; width: 100%; height: 100%; border-radius: 50%; overflow: hidden; box-shadow: 0 0 25px 8px rgba(200, 200, 255, 0.5), 0 0 50px 15px rgba(150, 150, 200, 0.3); }}
.moon-image {{ width: 100%; height: 100%; border-radius: 50%; object-fit: cover; }}
.moon-phase-shadow {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #0a0a1a; border-radius: 50%; transition: clip-path 0.5s ease; }}
.moon-glow {{ position: absolute; top: -20%; left: -20%; width: 140%; height: 140%; border-radius: 50%; background: radial-gradient(circle, rgba(200, 200, 255, 0.4) 0%, transparent 60%); pointer-events: none; animation: moonGlow 5s ease-in-out infinite; }}
@keyframes moonGlow {{ 0%, 100% {{ opacity: 0.7; }} 50% {{ opacity: 1; }} }}

/* --- AIRPLANE ANIMATION CSS --- */
.plane-icon-container {{ pointer-events: none; }}
#flying-plane {{ width: 100%; height: 100%; transition: transform 4s ease-out, filter 4s ease-out; }}
.plane-taking-off {{ transform: scale(0.3); filter: drop-shadow(0px 1px 1px rgba(0,0,0,0.9)); }}
.plane-flying {{ transform: scale(1.1); filter: drop-shadow(-15px 25px 12px rgba(0,0,0,0.6)); }}

</style>
</head>
<body class="{'forecast-mode' if is_forecast else 'live-mode'}">
<div id="loading-overlay">Initializing Map…</div>
<div id="map-container">
    <div id="map"></div>
    <div id="layer-selector">
        <label class="radio-label"><input type="radio" name="layerMode" value="pure_radar" checked onchange="setLayerMode('pure_radar')"> 📡</label>
        <label class="radio-label"><input type="radio" name="layerMode" value="radar" onchange="setLayerMode('radar')"> ☁️  </label>
        <label class="radio-label"><input type="radio" name="layerMode" value="temp" onchange="setLayerMode('temp')"> 🌍</label>
    </div>
    <div id="time-display">Loading...</div>
    <div id="left-controls">
        <button id="playBtn">▶</button>
        <div class="slider-container">
            <input type="range" id="slider" min="0" max="{max(0, len(radar_frames) - 1)}" value="0">
        </div>
    </div>
    {sun_html}
    {moon_html}
</div>
<script>
const frames = [{js_frames_array}];
const totalFrames = frames.length;
const isLiveMode = {str(not is_forecast).lower()};
let currentMode = 'pure_radar';

const activeBounds = {MAP_BOUNDS};
const viewBounds = [[24.0, -125.0], [50.0, -66.0]];
const erieLat = 42.1292;
const erieLon = -80.0851;

const map = L.map('map', {{
    zoomControl: false, minZoom: 4, maxZoom: 10, zoomSnap: 0,
    maxBounds: activeBounds, maxBoundsViscosity: 0.8
}});
map.fitBounds(viewBounds);
L.control.zoom({{ position: 'topright' }}).addTo(map);

// The deep ocean basemap setup
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ attribution: '&copy; Esri' }}).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ opacity: 0.4, attribution: '&copy; Esri' }}).addTo(map);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }}).addTo(map);

map.createPane('primaryPane'); map.getPane('primaryPane').style.zIndex = 410; map.getPane('primaryPane').classList.add('radar-blend');
let primaryLayer = L.imageOverlay('', activeBounds, {{pane: 'primaryPane', opacity: 0.85, interactive: false}}).addTo(map);

map.createPane('tempPane'); map.getPane('tempPane').style.zIndex = 420; map.getPane('tempPane').style.pointerEvents = 'none';
let tempOverlayLayer = L.imageOverlay('', activeBounds, {{pane: 'tempPane', opacity: 1.0, interactive: false}});

const tempLabelsGroup = L.layerGroup();
let labelMarkers = [];

const slider = document.getElementById('slider');
const timeDisplay = document.getElementById('time-display');
const playBtn = document.getElementById('playBtn');
const loadingOverlay = document.getElementById('loading-overlay');
const sunIndicator = document.getElementById('sun-indicator');
const moonIndicator = document.getElementById('moon-indicator');
const moonShadow = document.getElementById('moonShadow');

let timer = null;
let isPlaying = false;

function tempToColor(fahrenheit) {{
    let norm = Math.min(1, Math.max(0, (fahrenheit - 20) / 100.0));
    norm = Math.floor(norm * 10) / 10.0;
    let r = 0, g = 0, b = 0;
    if (norm < 0.25) {{ b = 139 + 116 * (norm / 0.25); }}
    else if (norm < 0.50) {{ g = 255 * ((norm - 0.25) / 0.25); b = 255 - 255 * ((norm - 0.25) / 0.25); }}
    else if (norm < 0.75) {{ r = 255 * ((norm - 0.50) / 0.25); g = 255 - 115 * ((norm - 0.50) / 0.25); }}
    else {{ r = 255 - 116 * ((norm - 0.75) / 0.25); g = 140 - 140 * ((norm - 0.75) / 0.25); }}
    return `rgb(${{Math.round(r)}}, ${{Math.round(g)}}, ${{Math.round(b)}})`;
}}

function updateLabels(gridData) {{
    if (labelMarkers.length === 0) {{
        gridData.forEach(pt => {{
            let icon = L.divIcon({{
                className: 'temp-label',
                html: `<span style="color:${{tempToColor(pt.val)}}">${{pt.val}}<span style="font-size: 0.15em;">/${{pt.hum}}</span></span>`,
                iconSize: [50, 20], iconAnchor: [25, 10]
            }});
            let marker = L.marker([pt.lat, pt.lon], {{icon: icon, interactive: false}});
            labelMarkers.push(marker);
            tempLabelsGroup.addLayer(marker);
        }});
    }} else {{
        gridData.forEach((pt, i) => {{
            if (labelMarkers[i]) labelMarkers[i].getElement().innerHTML = `<span style="color:${{tempToColor(pt.val)}}">${{pt.val}}<span style="font-size: 0.75em;">/${{pt.hum}}</span></span>`;
        }});
    }}
}}

function updateAstronomy(date) {{
    const sunPos = SunCalc.getPosition(date, erieLat, erieLon);
    const sunAltDegrees = sunPos.altitude * (180 / Math.PI);
    if (sunAltDegrees > -5) {{
        sunIndicator.style.opacity = Math.min(1, (sunAltDegrees + 5) / 15);
        sunIndicator.style.transform = `translateY(${{Math.max(0, 30 - sunAltDegrees)}}px)`;
    }} else {{ sunIndicator.style.opacity = 0; }}

    const moonPhaseInfo = SunCalc.getMoonIllumination(date);
    const phase = moonPhaseInfo.phase, fraction = moonPhaseInfo.fraction;
    let clipPath = '';
    if (fraction >= 0.99) clipPath = 'circle(0% at 50% 50%)';
    else if (fraction <= 0.01) clipPath = 'circle(100% at 50% 50%)';
    else {{
        const shadowWidth = (1 - fraction) * 100;
        clipPath = phase <= 0.5 ? `ellipse(${{shadowWidth}}% 100% at 0% 50%)` : `ellipse(${{shadowWidth}}% 100% at 100% 50%)`;
    }}
    moonShadow.style.clipPath = clipPath; moonShadow.style.webkitClipPath = clipPath;
}}

function drawFrame(index) {{
    if (!frames[index]) return;
    primaryLayer.setUrl(frames[index].radarImg);
    if (frames[index].tempImg && frames[index].tempImg.length > 50) {{
        tempOverlayLayer.setUrl(frames[index].tempImg);
        updateLabels(frames[index].tempGrid);
    }}
    timeDisplay.innerText = `${{frames[index].time}}`;
    updateAstronomy(new Date(frames[index].ts));
}}

function setLayerMode(mode) {{
    currentMode = mode;
    if (mode === 'pure_radar') {{ map.removeLayer(tempOverlayLayer); map.removeLayer(tempLabelsGroup); }}
    else if (mode === 'radar') {{ map.removeLayer(tempOverlayLayer); map.addLayer(tempLabelsGroup); }}
    else if (mode === 'temp') {{ map.addLayer(tempOverlayLayer); map.addLayer(tempLabelsGroup); }}
    drawFrame(slider.value);
}}

function nextFrame() {{ let n = parseInt(slider.value) + 1; if (n > totalFrames - 1) n = 0; slider.value = n; drawFrame(n); }}
function prevFrame() {{ let n = parseInt(slider.value) - 1; if (n < 0) n = totalFrames - 1; slider.value = n; drawFrame(n); }}

playBtn.onclick = () => {{
    if (isLiveMode) return;
    if (isPlaying) {{ clearInterval(timer); playBtn.innerHTML = "&#9654;"; isPlaying = false; }}
    else {{ timer = setInterval(nextFrame, 450); playBtn.innerHTML = "&#10074;&#10074;"; isPlaying = true; }}
}};
slider.oninput = (e) => {{ if (isLiveMode) return; if (isPlaying) playBtn.click(); drawFrame(e.target.value); }};

if (isLiveMode) {{ playBtn.innerHTML = "&#8987;"; playBtn.disabled = true; slider.disabled = true; }}
drawFrame(0);

// --- AIRPLANE ANIMATION JS ---
const planeStart = [33.94, -118.40]; // LAX coordinates
const planeEnd = [40.64, -73.77];    // JFK coordinates
const dy = planeEnd[0] - planeStart[0];
const dx = planeEnd[1] - planeStart[1];
const planeAngle = 90 - (Math.atan2(dy, dx) * 180 / Math.PI); // Orient plane toward destination

const planeSvg = `
<div style="transform: rotate(${{planeAngle}}deg); width: 100%; height: 100%;">
    <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'><defs><linearGradient id='gloss' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%23777777'/><stop offset='0.3' stop-color='%23151515'/><stop offset='1' stop-color='%23000000'/></linearGradient></defs><path fill='url(%23gloss)' stroke='%23333333' stroke-width='5' stroke-linejoin='round' d='M256,100 L16,300 L60,320 L130,240 L190,300 L230,250 L256,280 L282,250 L322,300 L382,240 L452,320 L496,300 Z'/></svg>" id="flying-plane" class="plane-taking-off" />
</div>`;

// Shrunk the icon slightly from [50, 50] to [36, 36] for better scale
// Shrunk the icon slightly from [50, 50] to [36, 36] for better scale
const planeIcon = L.divIcon({{ className: 'plane-icon-container', html: planeSvg, iconSize: [36, 36], iconAnchor: [18, 18] }});
const planeMarker = L.marker(planeStart, {{ icon: planeIcon, interactive: false, zIndexOffset: 9999 }}).addTo(map);

let flightStartTime = Date.now();
const flightDuration = 18000; // Takes 18 seconds to fly across the map

function animateFlight() {{
    let now = Date.now();
    let progress = (now - flightStartTime) / flightDuration;
    let planeEl = document.getElementById('flying-plane');

    if (progress > 1) {{
        flightStartTime = Date.now();
        progress = 0;
        if (planeEl) {{ planeEl.classList.remove('plane-flying'); planeEl.classList.add('plane-taking-off'); }}
    }} else if (progress > 0.08 && planeEl && planeEl.classList.contains('plane-taking-off')) {{
        planeEl.classList.remove('plane-taking-off'); planeEl.classList.add('plane-flying');
    }}

    planeMarker.setLatLng([planeStart[0] + dy * progress, planeStart[1] + dx * progress]);
    requestAnimationFrame(animateFlight);
}}
animateFlight();

map.whenReady(() => {{
    if (isLiveMode) setTimeout(() => {{ loadingOverlay.classList.add('hidden'); }}, 600);
    else setTimeout(() => {{ document.body.classList.add('loaded'); if (totalFrames > 1) playBtn.click(); }}, 450);
}});
</script>
</body>
</html>
"""

def render_flipbook():
    map_placeholder = st.empty()
    
    live_frame = fetch_live_frame()
    if live_frame:
        initial_html = generate_map_html(live_frame, mode="live")
        with map_placeholder:
            components.html(initial_html, height=850)
    else:
        st.error("Failed to fetch live radar imagery.")
        
    forecast_frames = fetch_forecast_frames()
    all_frames = live_frame + forecast_frames
    
    if len(all_frames) > 1:
        full_html = generate_map_html(all_frames, mode="forecast")
        with map_placeholder:
            components.html(full_html, height=850)

render_flipbook()
