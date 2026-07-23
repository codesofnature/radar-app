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
st.set_page_config(page_title="Instant Radar", layout="wide", page_icon="🌦️", initial_sidebar_state="collapsed")

# --- Streamlit UI Tweaks ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding: 0rem !important; max-width: 100% !important; }
.stApp { background-color: #cce4f0 !important; }
iframe { border: none; }
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

# --- Shared HTTP Session (connection pooling = massive speedup) ---
_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=30, max_retries=1)
_session.mount("https://", adapter)
_session.mount("http://", adapter)

# --- Precomputed Temperature Color LUT (eliminates per-pixel branching) ---
# Maps temperature index 0..180 (representing -60..120°F in 1° steps) → RGBA
_TEMP_LUT = np.zeros((181, 4), dtype=np.uint8)
for _i in range(181):
    _t = _i - 60  # actual temp in °F
    if _t < -20:
        _TEMP_LUT[_i] = [0, 0, 139, 45]
    elif _t < 60:
        _f = (_t + 20) / 80.0
        _TEMP_LUT[_i] = [
            int(np.clip(173 * _f, 0, 255)),
            int(np.clip(216 * _f, 0, 255)),
            int(np.clip(139 + 116 * _f, 0, 255)),
            45,
        ]
    elif _t < 82:
        _f = (_t - 60) / 22.0
        _TEMP_LUT[_i] = [
            int(np.clip(144 - 144 * _f, 0, 255)),
            int(np.clip(238 - 138 * _f, 0, 255)),
            int(np.clip(144 - 144 * _f, 0, 255)),
            45,
        ]
    else:
        # FIX: clamp _f to [0, 1] so values never exceed 255 or go below 0
        _f = max(0.0, min(1.0, (_t - 83) / 37.0))
        _TEMP_LUT[_i] = [
            int(np.clip(255 - 116 * _f, 0, 255)),
            int(np.clip(182 - 182 * _f, 0, 255)),
            int(np.clip(193 - 193 * _f, 0, 255)),
            45,
        ]

# --- NASA-Quality Embedded Images ---
@lru_cache(maxsize=1)
def get_embedded_sun_image():
    try:
        resp = _session.get("https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg", timeout=6)
        if resp.status_code == 200:
            return f"data:image/jpeg;base64,{base64.b64encode(resp.content).decode()}"
    except Exception:
        pass
    return ""

@lru_cache(maxsize=1)
def get_embedded_moon_image():
    try:
        resp = _session.get("https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg", timeout=8)
        if resp.status_code == 200:
            return f"data:image/jpeg;base64,{base64.b64encode(resp.content).decode()}"
    except Exception:
        pass
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
        lat_list, lon_list = [], []
        for lat in lats:
            for lon in lons:
                lat_list.append(f"{lat:.2f}")
                lon_list.append(f"{lon:.2f}")
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={','.join(lat_list)}&longitude={','.join(lon_list)}"
            f"&hourly=temperature_2m,relativehumidity_2m&temperature_unit=fahrenheit"
            f"&past_days=1&forecast_days=3"
        )
        resp = _session.get(url, timeout=12)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        logger.error(f"Open-Meteo request failed: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def generate_temp_overlay_cached(target_hour_str):
    """Cache temp overlay per hour. Fetches om_data internally (already cached)."""
    om_data = fetch_openmeteo_data()
    if not om_data:
        return "", []
    try:
        time_arr = om_data[0].get("hourly", {}).get("time", [])
        t_idx = -1
        try:
            t_idx = time_arr.index(target_hour_str)
        except ValueError:
            pass

        temps, grid_data, idx = [], [], 0
        for r, lat in enumerate(lats):
            for c, lon in enumerate(lons):
                val, hum = 50.0, 50.0
                if t_idx != -1 and idx < len(om_data):
                    loc_temps = om_data[idx].get("hourly", {}).get("temperature_2m", [])
                    loc_hums = om_data[idx].get("hourly", {}).get("relativehumidity_2m", [])
                    if t_idx < len(loc_temps) and loc_temps[t_idx] is not None:
                        val = loc_temps[t_idx]
                    if t_idx < len(loc_hums) and loc_hums[t_idx] is not None:
                        hum = loc_hums[t_idx]
                temps.append(val)
                grid_data.append({"lat": float(lat), "lon": float(lon), "val": round(val), "hum": round(hum)})
                idx += 1

        temp_grid = np.array(temps, dtype=np.float32).reshape((ROWS, COLS))
        grid_img = Image.fromarray(temp_grid, mode="F")
        large_grid_img = grid_img.resize((RADAR_W, RADAR_H), Image.BICUBIC)
        smooth_temps = np.array(large_grid_img)

        # Use precomputed LUT
        t_clipped = np.clip(smooth_temps, -60, 120)
        t_indices = (t_clipped + 60).astype(np.uint16)
        colors = _TEMP_LUT[t_indices].copy()  # .copy() so we can modify alpha below

        # Edge detection for isotherm contours
        temp_buckets = np.floor(t_clipped / 10.0)
        edge = np.zeros(t_clipped.shape, dtype=bool)
        edge[1:, :] |= temp_buckets[1:, :] != temp_buckets[:-1, :]
        edge[:-1, :] |= temp_buckets[:-1, :] != temp_buckets[1:, :]
        edge[:, 1:] |= temp_buckets[:, 1:] != temp_buckets[:, :-1]
        edge[:, :-1] |= temp_buckets[:, :-1] != temp_buckets[:, 1:]
        edge[2:, :] |= temp_buckets[2:, :] != temp_buckets[:-2, :]
        edge[:-2, :] |= temp_buckets[:-2, :] != temp_buckets[2:, :]
        edge[:, 2:] |= temp_buckets[:, 2:] != temp_buckets[:, :-2]
        edge[:, :-2] |= temp_buckets[:, :-2] != temp_buckets[:, 2:]

        colors[edge, 3] = 160
        final_img = Image.fromarray(colors, mode="RGBA")
        buf = io.BytesIO()
        final_img.save(buf, format="PNG", compress_level=1)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}", grid_data
    except Exception as e:
        logger.error(f"Temp overlay error: {e}")
        return "", []


def generate_temp_overlay(target_dt, om_data):
    """Wrapper: rounds to the hour, delegates to cached version."""
    target_hour = target_dt.replace(minute=0, second=0, microsecond=0)
    if target_dt.minute >= 30:
        target_hour += timedelta(hours=1)
    target_time_str = target_hour.strftime("%Y-%m-%dT%H:00")
    return generate_temp_overlay_cached(target_time_str)

def process_radar_image(img_bytes):
    """Optimized radar recoloring using vectorized numpy ops."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        data = np.asarray(img)
        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

        out = np.zeros_like(data)
        valid = a >= 100
        is_severe = valid & (r > 200) & (g < 80)
        is_mod = valid & (r > 200) & (g >= 80) & (g < 220) & (b < 100)
        is_light = valid & ~is_severe & ~is_mod

        out[is_severe] = [255, 0, 0, 255]
        out[is_mod] = [0, 0, 255, 255]
        out[is_light] = [0, 255, 255, 255]

        # Boost alpha
        new_a = np.where(a < 200, np.minimum(a.astype(np.int16) + 55, 255).astype(np.uint8), np.uint8(255))
        out[valid, 3] = new_a[valid]

        final_img = Image.fromarray(out, "RGBA")
        buf = io.BytesIO()
        final_img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return img_bytes

@st.cache_data(ttl=300, show_spinner=False)
def get_model_init_time():
    try:
        res = _session.get("https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/refd_1080.json", timeout=5)
        dt_str = res.json()["model_init_utc"]
        if not dt_str.endswith("Z"):
            dt_str += "Z"
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

def fetch_frame_data(url_info, om_data):
    """Fetch a single radar frame, process it, and attach temp overlay."""
    frame_time, time_str, url = url_info
    radar_src = None
    for attempt in range(3):
        try:
            resp = _session.get(url, timeout=10)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                processed_bytes = process_radar_image(resp.content)
                radar_src = f"data:image/png;base64,{base64.b64encode(processed_bytes).decode()}"
                break
            elif resp.status_code == 404:
                return None
        except Exception:
            if attempt < 2:
                time.sleep(0.3 * (attempt + 1))

    if radar_src:
        temp_img, temp_grid = generate_temp_overlay(frame_time, om_data)
        return {
            "dt": frame_time,
            "time": time_str,
            "radarImg": radar_src,
            "tempImg": temp_img,
            "tempGrid": temp_grid,
        }
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_opensky_planes():
    url = "https://opensky-network.org/api/states/all?lamin=24.0&lomin=-125.0&lamax=50.0&lomax=-66.0"
    headers = {"User-Agent": "InstantRadar/1.0 (contact: user@email.com)"}
    try:
        resp = _session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            states = resp.json().get("states", [])
            planes = []
            for s in states:
                if s[5] and s[6] and s[7] is not None and s[9] and s[10]:
                    cat = s[17] if len(s) > 17 else 0
                    is_large_category = cat in [4, 5, 6]
                    is_high_altitude = s[7] > 7000
                    is_fast = s[9] > 180
                    if is_large_category or is_high_altitude or is_fast:
                        planes.append({
                            "callsign": s[1].strip() if s[1] else "UNKNOWN",
                            "lon": s[5], "lat": s[6],
                            "altitude": s[7], "velocity": s[9], "heading": s[10],
                        })
            return planes[:150]
        elif resp.status_code == 429:
            logger.warning("OpenSky rate limited")
        return []
    except Exception as e:
        logger.error(f"OpenSky error: {e}")
        return []

def generate_map_html(radar_frames, mode="live", include_astronomy=True):
    is_forecast = mode == "forecast"
    all_planes = fetch_opensky_planes()
    planes_json = json.dumps(all_planes)

    init_time = get_model_init_time()
    first_ts_val = int(init_time.timestamp() * 1000)
    last_ts_val = int((init_time + timedelta(minutes=48 * 60)).timestamp() * 1000)

    sun_img_data = get_embedded_sun_image() if include_astronomy else ""
    moon_img_data = get_embedded_moon_image() if include_astronomy else ""

    frames_js_list = []
    for f in radar_frames:
        grid_json = json.dumps(f["tempGrid"])
        ts_val = int(f["dt"].timestamp() * 1000)
        frames_js_list.append(
            f"{{ ts: {ts_val}, time: '{f['time']}', radarImg: '{f['radarImg']}', tempImg: '{f['tempImg']}', tempGrid: {grid_json} }}"
        )
    js_frames_array = ",\n".join(frames_js_list)

    sun_html = f"""
    <div id="sun-indicator">
        <div class="sun-glow"></div>
        <div class="sun-image-container">
            <img class="sun-image" src="{sun_img_data}" alt="Sun">
        </div>
    </div>""" if include_astronomy else ""

    moon_html = f"""
    <div id="moon-indicator">
        <div class="moon-glow"></div>
        <div class="moon-image-container">
            <img class="moon-image" src="{moon_img_data}" alt="Moon">
            <div class="moon-phase-shadow" id="moonShadow"></div>
        </div>
    </div>""" if include_astronomy else ""

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Radar">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/suncalc/1.8.0/suncalc.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {{ margin:0; padding:0; background:transparent; font-family:-apple-system,BlinkMacSystemFont,sans-serif; overflow:hidden; position:fixed; width:100%; height:100dvh; }}
        #map-container {{ position:fixed; top:0; left:0; width:100vw; height:100dvh; }}
        #map {{ width:100%; height:100%; background:transparent; }}
        .radar-blend {{ mix-blend-mode:multiply; }}
        .radar-blend img {{ filter:drop-shadow(-10px 10px 8px rgba(0,0,0,0.5)); }}
        .temp-label {{ font-family:-apple-system,sans-serif; font-size:20px; font-weight:200; text-align:center; pointer-events:none; margin-top:-8px; text-shadow:0px 1px 2px rgba(0,0,0,0.8); }}
        .switch-container {{ position:absolute; top:50%; left:30px; transform:translateY(-50%); z-index:9999; width:64px; height:180px; background:rgba(255,255,255,0.15); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border-radius:32px; box-shadow:0 4px 20px rgba(0,0,0,0.1); border:1px solid rgba(255,255,255,0.4); overflow:hidden; }}
        #jelly-canvas {{ position:absolute; top:0; left:0; width:100%; height:100%; z-index:1; pointer-events:none; }}
        .ui-layer {{ position:absolute; top:0; left:0; width:100%; height:100%; z-index:2; display:flex; flex-direction:column; align-items:center; justify-content:space-between; padding:12px 0; box-sizing:border-box; }}
        .icon-btn {{ background:none; border:none; width:40px; height:40px; border-radius:50%; cursor:pointer; display:flex; align-items:center; justify-content:center; font-size:20px; z-index:3; padding:0; margin:0; outline:none; transition:transform 0.2s; }}
        .icon-btn:hover {{ transform:scale(1.1); }}
        #bottom-bar {{ position:fixed; top:16px; left:50%; transform:translateX(-50%); z-index:9999; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.15); backdrop-filter:blur(4px); border-radius:24px; padding:12px 24px 8px; display:flex; flex-direction:column; align-items:center; gap:4px; width:85vw; max-width:600px; }}
        #time-display {{ font-size:20px; font-weight:700; color:#fff; background-color:#ef4444; padding:8px 24px; min-width:200px; text-align:center; border-radius:999px; border:none; outline:none; box-shadow:none; margin-top:8px; margin-bottom:0; z-index:6; }}
        #slider-row {{ display:flex; align-items:center; gap:16px; width:100%; }}
        #playBtn {{ background:transparent; border:none; color:#888; width:44px; height:44px; border-radius:50%; flex-shrink:0; cursor:pointer; font-size:24px; display:flex; align-items:center; justify-content:center; transition:opacity 0.2s; box-shadow:none; }}
        #playBtn:hover:not(:disabled) {{ background:rgba(150,150,150,0.2); color:#555; }}
        #playBtn:disabled {{ background:transparent; color:#94a3b8; cursor:wait; }}
        .scrubber-container {{ position:relative; overflow:hidden; height:75px; width:100%; display:flex; align-items:center; mask-image:linear-gradient(to right,transparent,black 15%,black 85%,transparent); -webkit-mask-image:linear-gradient(to right,transparent,black 15%,black 85%,transparent); }}
        #fixed-playhead {{ position:absolute; left:50%; top:15%; height:85px; width:12px; border-radius:6px; background:#ef4444; transform:translate(-50%,-50%); box-shadow:none; z-index:5; pointer-events:none; }}
        #moving-track {{ position:absolute; left:50%; top:40%; width:250%; height:75px; transform:translateY(-50%); transition:transform 0.1s linear; z-index:2; pointer-events:none; }}
        #track-line {{ position:absolute; top:24px; left:0; width:100%; height:4px; background:rgba(0,0,0,0.20); border-radius:6px; }}
        #tick-row {{ position:absolute; top:24px; left:0; width:100%; height:50px; }}
        .slider-col {{ flex:1; display:flex; flex-direction:column; gap:0; }}
        input[type="range"] {{ -webkit-appearance:none; position:absolute; top:0; left:0; width:100%; height:100%; opacity:0; cursor:grab; z-index:10; margin:0; }}
        input[type="range"]:active {{ cursor:grabbing; }}
        .tk {{ position:absolute; top:0; width:1px; background:rgba(0,0,0,0.25); transform:translateX(-50%); }}
        .tk.maj {{ background:rgba(0,0,0,0.45); }}
        .tl {{ position:absolute; top:20px; font-size:14px; font-family:-apple-system,sans-serif; color:#333; font-weight:600; transform:translateX(-50%); white-space:nowrap; text-shadow:0px 1px 3px rgba(255,255,255,0.9); }}
        .tl.day {{ color:#000; font-weight:800; font-size:18px; top:18px; }}
        .sun-image[src=""], .moon-image[src=""] {{ visibility:hidden; }}
        .sun-image:not([src=""]), .moon-image:not([src=""]) {{ visibility:visible; }}
        #loading-overlay {{ position:absolute; top:80px; left:50%; transform:translateX(-50%); background:rgba(255,255,255,0.85); backdrop-filter:blur(8px); border-radius:20px; padding:10px 24px; z-index:10000; font-size:16px; font-weight:bold; color:#334155; box-shadow:0 4px 15px rgba(0,0,0,0.15); transition:opacity 0.4s ease-out; pointer-events:none; }}
        #loading-overlay.hidden {{ opacity:0; pointer-events:none; visibility:hidden; }}
        .leaflet-top.leaflet-right {{ top:15px; right:15px; }}
        #sun-indicator {{ position:absolute; bottom:25px; left:25px; z-index:9999; width:80px; height:80px; transition:opacity 0.5s ease,transform 0.5s ease; pointer-events:none; opacity:0; }}
        .sun-image-container {{ position:relative; width:100%; height:100%; border-radius:50%; overflow:hidden; animation:sunPulse 4s ease-in-out infinite; }}
        .sun-image {{ width:100%; height:100%; border-radius:50%; object-fit:cover; transform:scale(1.4); filter:drop-shadow(0 0 30px rgba(255,200,0,0.9)) drop-shadow(0 0 60px rgba(255,140,0,0.6)); }}
        .sun-glow {{ position:absolute; top:-30%; left:-30%; width:160%; height:160%; border-radius:50%; background:radial-gradient(circle,rgba(255,200,0,0.5),rgba(255,140,0,0.3) 40%,transparent 70%); animation:glowPulse 3s ease-in-out infinite; }}
        @keyframes sunPulse {{ 0%,100% {{ transform:scale(1) rotate(0deg); }} 50% {{ transform:scale(1.08) rotate(2deg); }} }}
        @keyframes glowPulse {{ 0%,100% {{ opacity:0.6; transform:scale(1); }} 50% {{ opacity:1; transform:scale(1.1); }} }}
        #moon-indicator {{ position:absolute; bottom:25px; right:25px; z-index:9999; width:80px; height:80px; transition:opacity 0.5s ease; pointer-events:none; }}
        .moon-image-container {{ position:relative; width:100%; height:100%; border-radius:50%; overflow:hidden; box-shadow:0 0 25px 8px rgba(200,200,255,0.5); }}
        .moon-image {{ width:100%; height:100%; border-radius:50%; object-fit:cover; }}
        .moon-phase-shadow {{ position:absolute; top:0; left:0; width:100%; height:100%; background:#0a0a1a; border-radius:50%; transition:clip-path 0.5s ease; }}
        .moon-glow {{ position:absolute; top:-20%; left:-20%; width:140%; height:140%; border-radius:50%; background:radial-gradient(circle,rgba(200,200,255,0.4),transparent 60%); animation:moonGlow 5s ease-in-out infinite; }}
        @keyframes moonGlow {{ 0%,100% {{ opacity:0.7; }} 50% {{ opacity:1; }} }}
        .plane-icon-container {{ pointer-events:none; }}
        .plane-wrapper {{ width:100%; height:100%; position:relative; display:flex; justify-content:center; align-items:center; }}
        .plane-shadow {{ position:absolute; left:50%; transform:translateX(-50%); width:140%; height:20%; background:radial-gradient(ellipse,rgba(0,0,0,0.55),rgba(0,0,0,0.2) 40%,transparent 70%); border-radius:50%; bottom:-28px; opacity:0.22; }}
        .dolphin-container {{ pointer-events:none; }}
        .dolphin-wrapper {{ width:100%; height:100%; position:relative; }}
        .dolphin-bobbing {{ width:100%; height:100%; animation:dolphinBob 2.5s ease-in-out infinite; }}
        .dolphin-shadow {{ position:absolute; left:50%; transform:translateX(-50%); width:120%; height:25%; background:radial-gradient(ellipse,rgba(0,30,80,0.4),transparent 60%); border-radius:50%; bottom:-15px; animation:dolphinShadow 2.5s ease-in-out infinite; }}
        @keyframes dolphinBob {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-6px); }} }}
        @keyframes dolphinShadow {{ 0%,100% {{ transform:translateX(-50%) scale(1); opacity:0.4; }} 50% {{ transform:translateX(-50%) scale(0.6); opacity:0.15; }} }}
        .dolphin-splash {{ position:absolute; bottom:0; left:50%; transform:translateX(-50%); width:34px; height:12px; background:radial-gradient(ellipse at center,rgba(255,255,255,0.75),rgba(255,255,255,0) 70%); border-radius:50%; animation:wakePulse 1.8s ease-in-out infinite; }}
        @keyframes wakePulse {{ 0%,100% {{ opacity:0.5; transform:translateX(-50%) scale(0.8); }} 50% {{ opacity:0.9; transform:translateX(-50%) scale(1.2); }} }}
        @media (max-width:768px) {{
            #bottom-bar {{ min-width:90vw; max-width:95vw; }}
            .tl {{ font-size:11px; }}
            .tl.day {{ font-size:18px; }}
        }}
    </style>
</head>
<body>
    <div id="map-container">
        <div id="map"></div>
        <div id="loading-overlay">Linking to NOAA satellites…</div>
        {sun_html}
        {moon_html}
        <div class="switch-container">
            <canvas id="jelly-canvas"></canvas>
            <div class="ui-layer">
                <button class="icon-btn active" data-idx="0" data-mode="pure_radar">🌤️</button>
                <button class="icon-btn" data-idx="1" data-mode="radar">🌡️</button>
                <button class="icon-btn" data-idx="2" data-mode="temp">🗺️</button>
            </div>
        </div>
        <div id="bottom-bar">
            <div id="time-display">Connecting to NOAA satellites…</div>
            <div id="slider-row">
                <button id="playBtn">▶</button>
                <div class="slider-col scrubber-container">
                    <div id="fixed-playhead"></div>
                    <div id="moving-track">
                        <div id="track-line"></div>
                        <div id="tick-row"></div>
                    </div>
                    <input type="range" id="slider" min="0" max="{max(0, len(radar_frames)-1)}" value="0" step="0.01">
                </div>
            </div>
        </div>
    </div>
    <script>
    (function() {{
        'use strict';
        let targetY = 0.0, headY = 0.0, tailY = 0.0, headVel = 0, tailVel = 0;

        function getUvY(el) {{
            const c = document.querySelector('.switch-container');
            const r = el.getBoundingClientRect(), cr = c.getBoundingClientRect();
            const py = (r.top - cr.top) + r.height / 2;
            const h = c.clientHeight, w = c.clientWidth;
            if (jMat) jMat.uniforms.u_aspect.value = h / w;
            return ((1.0 - py / h) - 0.5) * (h / w);
        }}

        function setTarget(i, mode, btn) {{
            targetY = getUvY(btn);
            document.querySelectorAll('.icon-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            setLayerMode(mode);
        }}

        // Wire buttons programmatically (works inside IIFE)
        document.querySelectorAll('.icon-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                setTarget(parseInt(this.dataset.idx), this.dataset.mode, this);
            }});
        }});

        const jCanvas = document.getElementById('jelly-canvas');
        const jRenderer = new THREE.WebGLRenderer({{ canvas: jCanvas, alpha: true, antialias: true }});
        jRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        jRenderer.setSize(64, 180);
        const jScene = new THREE.Scene();
        const jCamera = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
        let jMat = new THREE.ShaderMaterial({{
            uniforms: {{ u_headY:{{value:0}}, u_tailY:{{value:0}}, u_aspect:{{value:180/64}} }},
            vertexShader: `varying vec2 vUv; void main(){{ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }}`,
            fragmentShader: `
                varying vec2 vUv; uniform float u_headY,u_tailY,u_aspect;
                float sdCapsule(vec2 p,vec2 a,vec2 b,float r){{ vec2 pa=p-a,ba=b-a; float h=clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0); return length(pa-ba*h)-r; }}
                float map(vec2 p){{ return sdCapsule(p,vec2(0,u_headY),vec2(0,u_tailY),0.44); }}
                void main(){{
                    vec2 uv=vUv-0.5; uv.y*=u_aspect;
                    float d=map(uv);
                    float alpha=smoothstep(0.05,0.0,d);
                    vec2 eps=vec2(0.01,0.0);
                    vec3 n=normalize(vec3(map(uv+eps.xy)-map(uv-eps.xy),map(uv+eps.yx)-map(uv-eps.yx),0.08));
                    vec3 ld=normalize(vec3(-1,1,1.5));
                    float diff=max(dot(n,ld),0.0);
                    vec3 hd=normalize(ld+vec3(0,0,1));
                    float spec=pow(max(dot(n,hd),0.0),32.0);
                    vec3 col=mix(vec3(0.75,0.8,0.9),vec3(1),diff)+spec*0.8;
                    float ds=map(uv-vec2(0.02,-0.02));
                    float sa=smoothstep(0.15,0.0,ds)*0.35;
                    gl_FragColor=mix(vec4(0,0,0,sa),vec4(col,0.95),alpha);
                }}`,
            transparent: true
        }});
        jScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), jMat));
        function animateJelly() {{
            headVel = (headVel + (targetY - headY) * 0.04) * 0.85; headY += headVel;
            tailVel = (tailVel + (headY - tailY) * 0.03) * 0.80; tailY += tailVel;
            jMat.uniforms.u_headY.value = headY;
            jMat.uniforms.u_tailY.value = tailY;
            jRenderer.render(jScene, jCamera);
            requestAnimationFrame(animateJelly);
        }}
        setTimeout(() => {{
            const ab = document.querySelector('.icon-btn.active');
            if (ab) {{ const y = getUvY(ab); targetY = headY = tailY = y; }}
            animateJelly();
        }}, 50);

        const frames = [{js_frames_array}];
        const totalFrames = frames.length;
        const isLiveMode = {str(not is_forecast).lower()};
        const firstTs = {first_ts_val}, lastTs = {last_ts_val};
        let currentMode = 'pure_radar';
        const activeBounds = {MAP_BOUNDS};
        const isPhone = window.innerWidth <= 768;
        const viewBounds = isPhone ? [[33.69,-87.63],[41.88,-78.89]] : [[24.0,-125.0],[50.0,-66.0]];
        const erieLat = 42.1292, erieLon = -80.0851;

        const map = L.map('map', {{ zoomControl:false, minZoom:4, maxZoom:10, zoomSnap:0, maxBounds:activeBounds, maxBoundsViscosity:0.8 }});
        map.fitBounds(viewBounds);
        L.control.zoom({{ position:'topright' }}).addTo(map);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{attribution:'© Esri'}}).addTo(map);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{{z}}/{{y}}/{{x}}', {{opacity:0.4,attribution:'© Esri'}}).addTo(map);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png', {{attribution:'© CARTO'}}).addTo(map);

        map.createPane('primaryPane'); map.getPane('primaryPane').style.zIndex=410; map.getPane('primaryPane').classList.add('radar-blend');
        const BLANK='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
        let primaryLayer = L.imageOverlay(BLANK, activeBounds, {{pane:'primaryPane',opacity:0.85,interactive:false}}).addTo(map);
        map.createPane('tempPane'); map.getPane('tempPane').style.zIndex=420; map.getPane('tempPane').style.pointerEvents='none';
        let tempOverlayLayer = L.imageOverlay(BLANK, activeBounds, {{pane:'tempPane',opacity:1.0,interactive:false}});
        const tempLabelsGroup = L.layerGroup();
        let labelMarkers = [];

        const slider=document.getElementById('slider'), timeDisplay=document.getElementById('time-display'),
              playBtn=document.getElementById('playBtn'), sunInd=document.getElementById('sun-indicator'),
              moonInd=document.getElementById('moon-indicator'), moonShadow=document.getElementById('moonShadow'),
              tickRow=document.getElementById('tick-row');
        let timer=null, isPlaying=false;

        function tempToColor(f) {{
            let t=Math.max(-60,Math.min(120,f)),r,g,b;
            if(t<=50){{let p=(t+60)/110;r=255-255*p;g=255;b=255-255*p;}}
            else if(t<=80){{let p=(t-50)/30;r=255*p;g=255-255*p;b=0;}}
            else{{let p=(t-80)/40;r=255-127*p;g=0;b=0;}}
            return `rgb(${{Math.round(r)}},${{Math.round(g)}},${{Math.round(b)}})`;
        }}
        function updateLabels(gd) {{
            if(!labelMarkers.length) {{
                gd.forEach(pt => {{
                    let icon=L.divIcon({{className:'temp-label',html:`<span style="color:${{tempToColor(pt.val)}}">${{pt.val}}<span style="font-size:0.75em;">/${{pt.hum}}</span></span>`,iconSize:[50,20],iconAnchor:[25,10]}});
                    let m=L.marker([pt.lat,pt.lon],{{icon,interactive:false}});
                    labelMarkers.push(m); tempLabelsGroup.addLayer(m);
                }});
            }} else {{
                gd.forEach((pt,i) => {{
                    if(labelMarkers[i]){{let el=labelMarkers[i].getElement();if(el)el.innerHTML=`<span style="color:${{tempToColor(pt.val)}}">${{pt.val}}<span style="font-size:0.75em;">/${{pt.hum}}</span></span>`;}}
                }});
            }}
        }}
        function updateAstronomy(date) {{
            if(!sunInd) return;
            const sp=SunCalc.getPosition(date,erieLat,erieLon);
            const alt=sp.altitude*(180/Math.PI);
            if(alt>-5){{sunInd.style.opacity=Math.min(1,(alt+5)/15);sunInd.style.transform=`translateY(${{Math.max(0,30-alt)}}px)`;}}
            else sunInd.style.opacity=0;
            if(!moonShadow) return;
            const mi=SunCalc.getMoonIllumination(date);
            const ph=mi.phase,fr=mi.fraction;
            let cp='';
            if(fr>=0.99)cp='circle(0% at 50% 50%)';
            else if(fr<=0.01)cp='circle(100% at 50% 50%)';
            else{{const sw=(1-fr)*100;cp=ph<=0.5?`ellipse(${{sw}}% 100% at 0% 50%)`:`ellipse(${{sw}}% 100% at 100% 50%)`;}}
            moonShadow.style.clipPath=cp; moonShadow.style.webkitClipPath=cp;
        }}
        function drawFrame(idx, skipTrack) {{
            if(!frames||!frames[idx])return;
            primaryLayer.setUrl(frames[idx].radarImg||BLANK);
            if(frames[idx].tempImg&&frames[idx].tempImg.length>50){{
                tempOverlayLayer.setUrl(frames[idx].tempImg);
                updateLabels(frames[idx].tempGrid);
            }}
            timeDisplay.innerText=frames[idx].time||'—';
            updateAstronomy(new Date(frames[idx].ts||Date.now()));
            if(!skipTrack&&totalFrames>1&&lastTs>firstTs){{
                const pct=((frames[idx].ts-firstTs)/(lastTs-firstTs))*100;
                const tr=document.getElementById('moving-track');
                if(tr){{tr.style.transition='transform 0.3s linear';tr.style.transform='translate(-'+pct+'%,-50%)';}}
            }}
        }}
        if(!totalFrames){{timeDisplay.innerText="NOAA Satellites Locked...";}}

        function setLayerMode(mode) {{
            currentMode=mode;
            if(mode==='pure_radar'){{map.removeLayer(tempOverlayLayer);map.removeLayer(tempLabelsGroup);}}
            else if(mode==='radar'){{map.removeLayer(tempOverlayLayer);map.addLayer(tempLabelsGroup);}}
            else if(mode==='temp'){{map.addLayer(tempOverlayLayer);map.addLayer(tempLabelsGroup);}}
            drawFrame(Math.round(parseFloat(slider.value)));
        }}

        function nextFrame(){{let n=Math.round(parseFloat(slider.value))+1;if(n>=totalFrames)n=0;slider.value=n;drawFrame(n);}}
        playBtn.onclick=()=>{{
            if(isLiveMode)return;
            if(isPlaying){{clearInterval(timer);timer=null;playBtn.innerHTML="▶";isPlaying=false;}}
            else{{slider.value=Math.round(parseFloat(slider.value));timer=setInterval(nextFrame,450);playBtn.innerHTML="⏸";isPlaying=true;}}
        }};
        let lastDrawn=-1;
        slider.oninput=(e)=>{{
            if(isLiveMode)return;
            if(isPlaying)playBtn.click();
            const ev=parseFloat(e.target.value),iv=Math.round(ev);
            if(totalFrames>1&&lastTs>firstTs){{
                const li=Math.floor(ev),hi=Math.ceil(ev),rem=ev-li;
                const lts=frames[li]?frames[li].ts:firstTs, hts=frames[hi]?frames[hi].ts:lastTs;
                const its=lts+(hts-lts)*rem;
                const pct=((its-firstTs)/(lastTs-firstTs))*100;
                const tr=document.getElementById('moving-track');
                if(tr){{tr.style.transition='none';tr.style.transform='translate(-'+pct+'%,-50%)';}}
            }}
            if(iv!==lastDrawn&&frames[iv]){{drawFrame(iv,true);lastDrawn=iv;}}
        }};
        if(isLiveMode){{playBtn.innerHTML="";playBtn.disabled=true;slider.disabled=true;}}

        (function buildTicks(){{
            if(!tickRow)return;
            const span=lastTs-firstTs; if(span<=0)return;
            const minor=3600000;
            let t=Math.ceil(firstTs/minor)*minor;
            while(t<=lastTs){{
                const pct=((t-firstTs)/span)*100;
                const d=new Date(t),h=d.getHours();
                const isMaj=h%3===0;
                const tk=document.createElement('div');
                tk.className='tk'+(isMaj?' maj':'');
                tk.style.left=pct+'%'; tk.style.height=isMaj?'8px':'4px';
                tickRow.appendChild(tk);
                if(isMaj){{
                    const isDay=h===0;
                    const tl=document.createElement('div');
                    tl.className='tl'+(isDay?' day':'');
                    tl.style.left=pct+'%';
                    if(isDay)tl.textContent=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d.getDay()];
                    else{{let dh=h%12;if(!dh)dh=12;tl.textContent=dh+(h<12?'am':'pm');}}
                    tickRow.appendChild(tl);
                }}
                t+=minor;
            }}
        }})();
        drawFrame(0);

        // --- EMOJI PLANES ---
        const allPlanesCache={planes_json};
        let planeMarkers=[];
        function swapRandomPlanes(){{
            if(!allPlanesCache||!allPlanesCache.length)return;
            planeMarkers.forEach(pm=>map.removeLayer(pm.marker)); planeMarkers=[];
            const cb=map.getBounds();
            const vis=allPlanesCache.filter(p=>cb.contains(L.latLng(p.lat,p.lon)));
            const pool=vis.length?vis:allPlanesCache;
            const sel=[...pool].sort(()=>0.5-Math.random()).slice(0,5);
            sel.forEach((p,i)=>{{
                const ang=p.heading-65;
                const mph=Math.round(p.velocity*2.23694);
                const altK=Math.round(p.altitude*3.28084/1000);
                const html='<div class="plane-wrapper">'
                    +'<div style="position:absolute;top:-25px;background:rgba(255,255,255,0.05);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);padding:2px 8px;border-radius:6px;border:1px solid rgba(0,0,0,0.1);font-size:11px;font-weight:800;color:#0f172a;white-space:nowrap;pointer-events:none;z-index:10;">'+p.callsign+' \\u2022 '+mph+' mph \\u2022 '+altK+'k ft</div>'
                    +'<div class="plane-shadow"></div>'
                    +'<span style="font-size:32px;transform:rotate('+ang+'deg);filter:drop-shadow(3px 5px 4px rgba(0,0,0,0.4));line-height:1;">\\u2708\\uFE0F</span>'
                    +'</div>';
                const icon=L.divIcon({{className:'plane-icon-container',html,iconSize:[40,40],iconAnchor:[20,20]}});
                const marker=L.marker([p.lat,p.lon],{{icon,interactive:false,zIndexOffset:9999}}).addTo(map);
                planeMarkers.push({{marker,plane:p,startTime:Date.now()}});
            }});
        }}
        swapRandomPlanes(); setInterval(swapRandomPlanes,30000);
        function animatePlanes(){{
            const now=Date.now();
            planeMarkers.forEach(pm=>{{
                const es=(now-pm.startTime)/1000, dm=pm.plane.velocity*es, hr=pm.plane.heading*Math.PI/180;
                const dLat=dm*Math.cos(hr)/111320, dLon=dm*Math.sin(hr)/(111320*Math.cos(pm.plane.lat*Math.PI/180));
                pm.marker.setLatLng([pm.plane.lat+dLat,pm.plane.lon+dLon]);
            }});
            requestAnimationFrame(animatePlanes);
        }}
        setTimeout(animatePlanes,1000);

        // --- EMOJI DOLPHINS ---
        const dolphinPos=[
            {{center:[27.0,-88.0],rLat:1.5,rLon:1.0,cycle:60000}},
            {{center:[32.0,-65.5],rLat:1.2,rLon:1.8,cycle:55000}},
            {{center:[32.5,-125.5],rLat:2.5,rLon:1.2,cycle:65000}},
            {{center:[29.0,-78.5],rLat:1.5,rLon:1.0,cycle:45000}}
        ];
        const dolphinMarkers=[];
        dolphinPos.forEach((pos,idx)=>{{
            const html='<div class="dolphin-wrapper">'
                +'<div class="dolphin-shadow"></div>'
                +'<div class="dolphin-bobbing">'
                +'<div class="dolphin-splash"></div>'
                +'<span style="font-size:22px;filter:drop-shadow(2px 3px 2px rgba(0,0,0,0.5));line-height:1;">\\uD83D\\uDC2C</span>'
                +'</div></div>';
            const icon=L.divIcon({{className:'dolphin-container',html,iconSize:[20,20],iconAnchor:[10,10]}});
            const marker=L.marker(pos.center,{{icon,interactive:false,zIndexOffset:800}}).addTo(map);
            dolphinMarkers.push({{marker,pos,startTime:Date.now()-idx*15000}});
        }});
        function animateDolphins(){{
            dolphinMarkers.forEach(dm=>{{
                const el=Date.now()-dm.startTime, t=(el%dm.pos.cycle)/dm.pos.cycle, a=t*2*Math.PI;
                dm.marker.setLatLng([dm.pos.center[0]+dm.pos.rLat*Math.sin(a),dm.pos.center[1]+dm.pos.rLon*Math.cos(a)]);
            }});
            requestAnimationFrame(animateDolphins);
        }}
        setTimeout(animateDolphins,1500);

        setTimeout(()=>{{
            const ov=document.getElementById('loading-overlay');
            if(ov){{ov.style.opacity='0';setTimeout(()=>ov.remove(),500);}}
            if(!isLiveMode&&totalFrames>1&&!isPlaying){{const b=document.getElementById('playBtn');if(b)b.click();}}
        }},800);
    }})();
    </script>
</body>
</html>"""


def render_flipbook():
    """Single-pass render: fetch everything in parallel, render once."""
    map_placeholder = st.empty()

    # Gather all data in parallel
    om_data = fetch_openmeteo_data()
    init_time = get_model_init_time()
    now = datetime.now(timezone.utc)

    # Build URL list for forecast frames
    urls_to_fetch = []
    for mins_offset in MINUTES_OFFSETS:
        frame_time = init_time + timedelta(minutes=mins_offset)
        if frame_time > now:
            layer_name = f"refd_{str(mins_offset).zfill(4)}"
            url = (
                f"https://mesonet.agron.iastate.edu/cgi-bin/wms/hrrr/refd.cgi"
                f"?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer_name}"
                f"&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}"
                f"&CRS=EPSG:3857&BBOX={BBOX}"
            )
            local_time = frame_time.astimezone(LOCAL_TZ)
            time_str = local_time.strftime("%a, %b %d - %I:%M %p")
            urls_to_fetch.append((frame_time, time_str, url))

    # Also fetch the live NEXRAD frame
    live_url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi"
        f"?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0q-900913"
        f"&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}"
        f"&SRS=EPSG:3857&BBOX={BBOX}"
    )
    live_label = now.astimezone(LOCAL_TZ).strftime("%b %d - %I:%M %p (Live)")
    urls_to_fetch.insert(0, (now, live_label, live_url))

    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.markdown(
        "<div style='font-size:22px;font-weight:bold;margin-bottom:10px;'>Linking to NOAA satellites…</div>",
        unsafe_allow_html=True,
    )

    accumulated = []
    total = len(urls_to_fetch)
    completed = 0

    # Fetch ALL frames in parallel (live + forecast together)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_map = {
            executor.submit(fetch_frame_data, url_info, om_data): url_info
            for url_info in urls_to_fetch
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            if result:
                accumulated.append(result)
            completed += 1
            progress_bar.progress(min(completed / total, 1.0))

    progress_bar.empty()
    status_text.empty()

    # Sort chronologically and render ONCE
    accumulated.sort(key=lambda x: x["dt"])

    if accumulated:
        full_html = generate_map_html(accumulated, mode="forecast", include_astronomy=True)
        with map_placeholder:
            components.html(full_html, height=850, scrolling=False)
    else:
        map_placeholder.error("Failed to load radar data. Please refresh.")


if __name__ == "__main__":
    render_flipbook()
