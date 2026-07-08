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
_MOON_IMAGE_B64 = (
    "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"
)

@lru_cache(maxsize=None)
def get_embedded_moon_image():
    """Returns the embedded moon image as a data URI (no network required)."""
    return f"data:image/jpeg;base64,{_MOON_IMAGE_B64}"

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
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0.2px;
    text-align: center;
    pointer-events: none;
    margin-top: -8px;
    
    /* Softer, tighter shadow to support thinner text without blurring it */
    text-shadow: 
        0px 1px 2px rgba(0, 0, 0, 0.8),
        0px 0px 3px rgba(0, 0, 0, 0.5);
}}
#layer-selector {{
    position: absolute;
    left: 15px;
    top: 15px;
    z-index: 9999;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 12px 18px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    display: flex;
    flex-direction: column;
    gap: 12px;
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
}}
.radio-label {{
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
}}
.radio-label input[type="radio"] {{
    accent-color: #4f46e5;
    cursor: pointer;
    width: 16px;
    height: 16px;
}}
#time-display {{
    position: absolute;
    top: 15px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 10px 24px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    white-space: nowrap;
    letter-spacing: -0.5px;
}}
#left-controls {{
    position: absolute;
    left: 15px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 9999;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 15px 12px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 25px;
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
#sun-indicator {{
    position: absolute;
    bottom: 25px;
    left: 25px;
    z-index: 9999;
    width: 120px;
    height: 120px;
    transition: opacity 0.5s ease, transform 0.5s ease;
    pointer-events: none;
    opacity: 0;
}}
.sun-image-container {{
    position: relative;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    overflow: hidden;
    animation: sunPulse 4s ease-in-out infinite;
}}
.sun-image {{
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    transform: scale(1.4);
    filter: drop-shadow(0 0 30px rgba(255, 200, 0, 0.9))
            drop-shadow(0 0 60px rgba(255, 140, 0, 0.6))
            drop-shadow(0 0 90px rgba(255, 69, 0, 0.4));
}}
.sun-glow {{
    position: absolute;
    top: -30%;
    left: -30%;
    width: 160%;
    height: 160%;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 200, 0, 0.5) 0%, rgba(255, 140, 0, 0.3) 40%, transparent 70%);
    animation: glowPulse 3s ease-in-out infinite;
    pointer-events: none;
}}
@keyframes sunPulse {{
    0%, 100% {{ transform: scale(1) rotate(0deg); }}
    50% {{ transform: scale(1.08) rotate(2deg); }}
}}
@keyframes glowPulse {{
    0%, 100% {{ opacity: 0.6; transform: scale(1); }}
    50% {{ opacity: 1; transform: scale(1.1); }}
}}

/* --- NASA-Quality Moon Indicator --- */
#moon-indicator {{
    position: absolute;
    bottom: 25px;
    right: 25px;
    z-index: 9999;
    width: 100px;
    height: 100px;
    transition: opacity 0.5s ease;
    pointer-events: none;
}}
.moon-image-container {{
    position: relative;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    overflow: hidden;
    box-shadow: 0 0 25px 8px rgba(200, 200, 255, 0.5),
                0 0 50px 15px rgba(150, 150, 200, 0.3);
}}
.moon-image {{
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
}}
.moon-phase-shadow {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: #0a0a1a;
    border-radius: 50%;
    transition: clip-path 0.5s ease;
}}
.moon-glow {{
    position: absolute;
    top: -20%;
    left: -20%;
    width: 140%;
    height: 140%;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(200, 200, 255, 0.4) 0%, transparent 60%);
    pointer-events: none;
    animation: moonGlow 5s ease-in-out infinite;
}}
@keyframes moonGlow {{
    0%, 100% {{ opacity: 0.7; }}
    50% {{ opacity: 1; }}
}}
</style>
</head>
<body class="{'forecast-mode' if is_forecast else 'live-mode'}">
<div id="loading-overlay">Initializing Map…</div>
<div id="map-container">
    <div id="map"></div>
    <div id="layer-selector">
        <label class="radio-label">
            <input type="radio" name="layerMode" value="radar" onchange="toggleTemp(false)"> ⚡ Radar
        </label>
        <label class="radio-label">
            <input type="radio" name="layerMode" value="temp" checked onchange="toggleTemp(true)"> 🌡️ Radar + Temps
        </label>
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
let showTemps = true;

const activeBounds = {MAP_BOUNDS};
const viewBounds = [[24.0, -125.0], [50.0, -66.0]];
const erieLat = 42.1292;
const erieLon = -80.0851;

const map = L.map('map', {{
    zoomControl: false,
    minZoom: 4,
    maxZoom: 10,
    zoomSnap: 0,
    maxBounds: activeBounds,
    maxBoundsViscosity: 0.8
}});
map.fitBounds(viewBounds);
L.control.zoom({{ position: 'topright' }}).addTo(map);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }}).addTo(map);

map.createPane('primaryPane');
map.getPane('primaryPane').style.zIndex = 410;
map.getPane('primaryPane').classList.add('radar-blend');
let primaryLayer = L.imageOverlay('', activeBounds, {{pane: 'primaryPane', opacity: 0.85, interactive: false}}).addTo(map);

map.createPane('tempPane');
map.getPane('tempPane').style.zIndex = 420;
map.getPane('tempPane').style.pointerEvents = 'none';
let tempOverlayLayer = L.imageOverlay('', activeBounds, {{pane: 'tempPane', opacity: 1.0, interactive: false}});

const tempLabelsGroup = L.layerGroup();
let labelMarkers = [];

if (showTemps) {{
    tempOverlayLayer.addTo(map);
    tempLabelsGroup.addTo(map);
}}

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
    if (norm < 0.25) {{
        const f1 = norm / 0.25;
        b = 139 + 116 * f1;
    }} else if (norm < 0.50) {{
        const f2 = (norm - 0.25) / 0.25;
        g = 255 * f2;
        b = 255 - 255 * f2;
    }} else if (norm < 0.75) {{
        const f3 = (norm - 0.50) / 0.25;
        r = 255 * f3;
        g = 255 - 115 * f3;
    }} else {{
        const f4 = (norm - 0.75) / 0.25;
        r = 255 - 116 * f4;
        g = 140 - 140 * f4;
    }}
    return `rgb(${{Math.round(r)}}, ${{Math.round(g)}}, ${{Math.round(b)}})`;
}}

function updateLabels(gridData) {{
    if (labelMarkers.length === 0) {{
        gridData.forEach(pt => {{
            let icon = L.divIcon({{
                className: 'temp-label',
                html: `<span style="color:${{tempToColor(pt.val)}}">${{pt.val}}/${{pt.hum}}</span>`,
                iconSize: [50, 20],
                iconAnchor: [25, 10]
            }});
            let marker = L.marker([pt.lat, pt.lon], {{icon: icon, interactive: false}});
            labelMarkers.push(marker);
            tempLabelsGroup.addLayer(marker);
        }});
    }} else {{
        gridData.forEach((pt, i) => {{
            if (labelMarkers[i]) {{
                labelMarkers[i].getElement().innerHTML = `<span style="color:${{tempToColor(pt.val)}}">${{pt.val}}/${{pt.hum}}</span>`;
            }}
        }});
    }}
}}

function updateAstronomy(date) {{
    const sunPos = SunCalc.getPosition(date, erieLat, erieLon);
    const sunAltDegrees = sunPos.altitude * (180 / Math.PI);
    if (sunAltDegrees > -5) {{
        sunIndicator.style.opacity = Math.min(1, (sunAltDegrees + 5) / 15);
        const yOffset = Math.max(0, 30 - sunAltDegrees);
        sunIndicator.style.transform = `translateY(${{yOffset}}px)`;
    }} else {{
        sunIndicator.style.opacity = 0;
    }}

    const moonPhaseInfo = SunCalc.getMoonIllumination(date);
    const phase = moonPhaseInfo.phase;
    const fraction = moonPhaseInfo.fraction;
    let clipPath = '';
    if (fraction >= 0.99) {{
        clipPath = 'circle(0% at 50% 50%)';
    }} else if (fraction <= 0.01) {{
        clipPath = 'circle(100% at 50% 50%)';
    }} else {{
        const shadowWidth = (1 - fraction) * 100;
        if (phase <= 0.5) {{
            clipPath = `ellipse(${{shadowWidth}}% 100% at 0% 50%)`;
        }} else {{
            clipPath = `ellipse(${{shadowWidth}}% 100% at 100% 50%)`;
        }}
    }}
    moonShadow.style.clipPath = clipPath;
    moonShadow.style.webkitClipPath = clipPath;
}}

function drawFrame(index) {{
    if (!frames[index]) return;
    primaryLayer.setUrl(frames[index].radarImg);
    if (showTemps && frames[index].tempImg.length > 50) {{
        tempOverlayLayer.setUrl(frames[index].tempImg);
        updateLabels(frames[index].tempGrid);
    }}
    timeDisplay.innerText = `${{frames[index].time}}`;
    const frameDate = new Date(frames[index].ts);
    updateAstronomy(frameDate);
}}

function toggleTemp(show) {{
    showTemps = show;
    if (show) {{
        map.addLayer(tempOverlayLayer);
        map.addLayer(tempLabelsGroup);
    }} else {{
        map.removeLayer(tempOverlayLayer);
        map.removeLayer(tempLabelsGroup);
    }}
    drawFrame(slider.value);
}}

function nextFrame() {{
    let n = parseInt(slider.value) + 1;
    if (n > totalFrames - 1) n = 0;
    slider.value = n;
    drawFrame(n);
}}

function prevFrame() {{
    let n = parseInt(slider.value) - 1;
    if (n < 0) n = totalFrames - 1;
    slider.value = n;
    drawFrame(n);
}}

playBtn.onclick = () => {{
    if (isLiveMode) return;
    if (isPlaying) {{
        clearInterval(timer);
        playBtn.innerHTML = "&#9654;";
        isPlaying = false;
    }} else {{
        timer = setInterval(nextFrame, 450);
        playBtn.innerHTML = "&#10074;&#10074;";
        isPlaying = true;
    }}
}};

slider.oninput = (e) => {{
    if (isLiveMode) return;
    if (isPlaying) playBtn.click();
    drawFrame(e.target.value);
}};

if (isLiveMode) {{
    playBtn.innerHTML = "&#8987;";
    playBtn.disabled = true;
    slider.disabled = true;
}}

drawFrame(0);

map.whenReady(() => {{
    if (isLiveMode) {{
        setTimeout(() => {{ loadingOverlay.classList.add('hidden'); }}, 600);
    }} else {{
        setTimeout(() => {{
            document.body.classList.add('loaded');
            if (totalFrames > 1) playBtn.click();
        }}, 450);
    }}
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
