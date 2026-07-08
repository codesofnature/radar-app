import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import concurrent.futures
import math
import json
import os
import time
import logging
import xarray as xr
import numpy as np
from PIL import Image
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Instant Radar Flipbook", layout="wide", page_icon="⚡")

# --- Configuration ---
LOCAL_TZ = ZoneInfo("America/New_York")
USE_STATIC_SERVING = True
STATIC_DIR = "static/radar_frames"
GFS_DIR = "static/gfs_frames"
GFS_META_FILE = os.path.join(GFS_DIR, "meta.json")

if USE_STATIC_SERVING:
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(GFS_DIR, exist_ok=True)

BBOX = "-14200000,2700000,-7200000,6400000"
WIDTH = 1200
HEIGHT = 700
RADAR_RES_FACTOR = 3
RADAR_W = int(WIDTH * RADAR_RES_FACTOR)
RADAR_H = int(HEIGHT * RADAR_RES_FACTOR)

MINUTES_OFFSETS = list(range(0, 18 * 60, 15)) + list(range(18 * 60, 49 * 60, 60))

# --- Helpers ---
def mercator_to_latlon(x, y):
    r = 6378137.0
    lon = math.degrees(x / r)
    lat = math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2.0)
    return lat, lon

xmin, ymin, xmax, ymax = map(float, BBOX.split(","))
lat_min, lon_min = mercator_to_latlon(xmin, ymin)
lat_max, lon_max = mercator_to_latlon(xmax, ymax)
MAP_BOUNDS = f"[[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]]"
GFS_BOUNDS = "[[-85.0, -180.0], [85.0, 180.0]]"

@st.cache_data(ttl=300)
def get_model_init_time():
    try:
        res = requests.get("https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/refd_1080.json", timeout=5)
        dt_str = res.json()["model_init_utc"]
        if not dt_str.endswith("Z"): dt_str += "Z"
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.warning(f"Error fetching model init time: {e}")
        now = datetime.now(timezone.utc)
        return now.replace(minute=0, second=0, microsecond=0)

def fetch_single_image(url_info, max_retries=2):
    frame_time, time_str, url = url_info
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                if USE_STATIC_SERVING:
                    filename = f"frame_{int(frame_time.timestamp())}.png"
                    filepath = os.path.join(STATIC_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    return f"app/static/radar_frames/{filename}"
                else:
                    b64 = base64.b64encode(resp.content).decode("utf-8")
                    return f"data:image/png;base64,{b64}"
            elif resp.status_code == 404:
                return None
        except Exception:
            if attempt < max_retries: time.sleep(0.5 * (attempt + 1))
    return None

@st.cache_data(ttl=300, show_spinner=False)
def build_rainviewer_frames():
    try:
        data = requests.get('https://api.rainviewer.com/public/weather-maps.json', timeout=10).json()
        latest_frame = data['radar']['past'][-1] 
        dt = datetime.fromtimestamp(latest_frame['time'], timezone.utc)
        local_time = dt.astimezone(LOCAL_TZ)
        
        return [{
            "dt": dt,
            "time": local_time.strftime("🌍 Current Global Radar: %A, %I:%M %p"),
            "img": "rv_native",
            "rv_path": data['host'] + latest_frame['path']
        }]
    except Exception as e:
        logger.error(f"Error fetching Rainviewer: {e}")
        return []

@st.cache_data(ttl=300, show_spinner=False)
def build_hrrr_assets():
    init_time = get_model_init_time()
    now = datetime.now(timezone.utc)
    frames_data = []
    
    live_wms_url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0q-900913&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&SRS=EPSG:3857&BBOX={BBOX}"
    live_label = now.astimezone(LOCAL_TZ).strftime("🔴 USA NEXRAD: %A, %I:%M %p")
    live_img = fetch_single_image((now, live_label, live_wms_url))
    if live_img: frames_data.append({"dt": now, "time": live_label, "img": live_img})

    for attempt in range(3):
        hrrr_frames = []
        urls_to_fetch = []
        for mins_offset in MINUTES_OFFSETS:
            frame_time = init_time + timedelta(minutes=mins_offset)
            if frame_time > now:
                layer_name = f"refd_{str(mins_offset).zfill(4)}"
                url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/hrrr/refd.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer_name}&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&CRS=EPSG:3857&BBOX={BBOX}"
                local_time = frame_time.astimezone(LOCAL_TZ)
                time_str = local_time.strftime("🔮 USA HRRR: %A, %I:%M %p")
                urls_to_fetch.append((frame_time, time_str, url))

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(fetch_single_image, u): u for u in urls_to_fetch}
            for future in concurrent.futures.as_completed(futures):
                u_info = futures[future]
                img_src = future.result()
                if img_src: hrrr_frames.append({"dt": u_info[0], "time": u_info[1], "img": img_src})

        if len(hrrr_frames) > 5:
            hrrr_frames.sort(key=lambda x: x["dt"])
            frames_data.extend(hrrr_frames)
            return frames_data
        
        logger.warning(f"HRRR run {init_time} incomplete. Falling back...")
        init_time = init_time - timedelta(hours=1)

    return frames_data

def get_latest_gfs_run():
    now = datetime.now(timezone.utc)
    possible_runs = [0, 6, 12, 18]
    run_hr = 0
    for r in possible_runs:
        if now.hour >= r + 4: run_hr = r
        
    if now.hour < 4:
        run_date = now - timedelta(days=1)
        run_hr = 18
    else:
        run_date = now
        
    return run_date.strftime("%Y%m%d"), run_hr

def build_gfs_frames():
    os.makedirs(GFS_DIR, exist_ok=True)
    if os.path.exists(GFS_META_FILE):
        with open(GFS_META_FILE, 'r') as f:
            meta = json.load(f)
        cache_time = datetime.strptime(meta['fetch_time'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - cache_time < timedelta(hours=6):
            logger.info("Using cached GFS frames")
            frames = []
            for file in sorted(os.listdir(GFS_DIR)):
                if file.endswith(".png"):
                    parts = file.replace(".png", "").split("_")
                    date_str, run_hr, f_hr = parts[1], int(parts[2]), int(parts[3][1:])
                    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(hours=run_hr + f_hr)
                    frames.append({
                        "dt": dt,
                        "time": dt.strftime("🔮 GFS Forecast: %a, %b %d, %I:%M %p"),
                        "img": f"app/static/gfs_frames/{file}"
                    })
            if len(frames) > 10: return frames

    date_str, run_hr = get_latest_gfs_run()
    logger.info(f"Downloading GFS data for {date_str} run {run_hr}z")

    for file in os.listdir(GFS_DIR):
        if file.endswith(".png"): os.remove(os.path.join(GFS_DIR, file))
        
    frames = []
    fcst_hours = list(range(0, 385, 12)) 
    progress_bar = st.progress(0, text="Starting GFS download...")

    for i, f_hr in enumerate(fcst_hours):
        progress_bar.progress((i + 1) / len(fcst_hours), text=f"Downloading & Processing GFS frame {i+1}/{len(fcst_hours)} (f{f_hr:03d})...")
        
        url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        params = {
            "file": f"gfs.t{run_hr:02d}z.pgrb2.0p25.f{f_hr:03d}",
            "lev_surface": "on",
            "var_PRATE": "on", 
            "leftlon": "0", "rightlon": "360", "toplat": "90", "bottomlat": "-90",
            "dir": f"/gfs.{date_str}/{run_hr:02d}/atmos"
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200 or len(resp.content) < 1000: continue
                
            temp_grib = os.path.join(GFS_DIR, f"temp_{f_hr}.grib2")
            with open(temp_grib, 'wb') as f: f.write(resp.content)
                
            ds = xr.open_dataset(temp_grib, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'shortName': 'prate'}})
            prate = ds['prate'].values * 3600.0  
            prate = np.nan_to_num(prate, nan=0.0) 
            
            prate_shifted = np.roll(prate, shift=prate.shape[1]//2, axis=1)
            img_data = np.zeros((prate_shifted.shape[0], prate_shifted.shape[1], 4), dtype=np.uint8)
            
            img_data[prate_shifted < 0.1] = [0, 0, 0, 0]                                   
            img_data[(prate_shifted >= 0.1) & (prate_shifted < 1.0)] = [0, 255, 0, 160]    
            img_data[(prate_shifted >= 1.0) & (prate_shifted < 2.5)] = [255, 200, 0, 180]  
            img_data[prate_shifted >= 2.5] = [255, 0, 0, 220]                              
            
            img = Image.fromarray(img_data, 'RGBA')
            fname = f"gfs_{date_str}_{run_hr:02d}_f{f_hr:03d}.png"
            img.save(os.path.join(GFS_DIR, fname))
            
            dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(hours=run_hr + f_hr)
            frames.append({
                "dt": dt,
                "time": dt.strftime("🔮 GFS Forecast: %a, %b %d, %I:%M %p"),
                "img": f"app/static/gfs_frames/{fname}"
            })
            os.remove(temp_grib)
            
        except Exception as e:
            logger.error(f"Error processing GFS f{f_hr}: {e}")
            
    progress_bar.empty()

    if frames:
        with open(GFS_META_FILE, 'w') as f:
            json.dump({"fetch_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}, f)
    return frames

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_build_temp_grid(target_dts, lat_min_bound, lat_max_bound, lon_min_bound, lon_max_bound):
    points = []
    PAD = 0.0 # Strict zero padding. We do not want to exceed map limits!
    
    # Global dynamic spacing. Kept slightly wider to prevent DOM lag on 16 days of data.
    LAT_STEP = 15.0
    LON_STEP = 20.0

    # Explicit safety bounding to ensure Open-Meteo accepts our longitudes and latitudes.
    grid_lat_min = max(-90.0, lat_min_bound - PAD)
    grid_lat_max = min(90.0, lat_max_bound + PAD)
    grid_lon_min = max(-180.0, lon_min_bound - PAD)
    grid_lon_max = min(180.0, lon_max_bound + PAD)

    lat = grid_lat_min
    while lat <= grid_lat_max:
        lon = grid_lon_min
        while lon <= grid_lon_max:
            points.append({"lat": round(lat, 2), "lon": round(lon, 2)})
            lon += LON_STEP
        lat += LAT_STEP

    chunk_size = 40
    all_hourly_data = []

    for i in range(0, len(points), chunk_size):
        chunk = points[i : i + chunk_size]
        lats = ", ".join(str(p["lat"]) for p in chunk)
        lons = ", ".join(str(p["lon"]) for p in chunk)
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&hourly=temperature_2m&temperature_unit=fahrenheit&past_days=1&forecast_days=16&timezone=UTC"
        
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list): data = [data]
                
                if len(data) != len(chunk):
                    logger.error(f"Open-Meteo length mismatch. API likely rejected coordinates.")
                    all_hourly_data.extend([{} for _ in chunk])
                else:
                    all_hourly_data.extend(data)
            else:
                logger.error(f"Open-Meteo API Error: {resp.status_code} - {resp.text}")
                all_hourly_data.extend([{} for _ in chunk])
        except Exception as e:
            logger.warning(f"Error fetching Open-Meteo chunk {i}: {e}")
            all_hourly_data.extend([{} for _ in chunk])
            
        time.sleep(0.2)

    parsed_hourly = []
    for i in range(len(points)):
        h = all_hourly_data[i].get("hourly") if i < len(all_hourly_data) else None
        if not h:
            parsed_hourly.append(None)
            continue
        times_ts = [datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp() for t in h["time"]]
        parsed_hourly.append({"times": times_ts, "temps": h["temperature_2m"]})

    frames_grids = []
    for target_dt in target_dts:
        target_ts = target_dt.timestamp()
        grid = []
        for i, point in enumerate(points):
            ph = parsed_hourly[i]
            if not ph: continue
            times_ts = ph["times"]
            temps = ph["temps"]

            temp_val = None
            if target_ts <= times_ts[0]: temp_val = temps[0]
            elif target_ts >= times_ts[-1]: temp_val = temps[-1]
            else:
                lo, hi = 0, len(times_ts) - 1
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if times_ts[mid] <= target_ts: lo = mid
                    else: hi = mid
                t0, t1 = times_ts[lo], times_ts[hi]
                v0, v1 = temps[lo], temps[hi]
                if v0 is not None and v1 is not None:
                    temp_val = v0 + (v1 - v0) * ((target_ts - t0) / (t1 - t0))

            if temp_val is not None:
                grid.append({"lat": point["lat"], "lon": point["lon"], "t": int(round(temp_val))})
        frames_grids.append(grid)

    return frames_grids

# --- Flipbook Renderer ---
def render_flipbook(mode):
    radar_frames = []
    active_bounds = MAP_BOUNDS if mode == "hrrr_only" else GFS_BOUNDS

    if mode == "hrrr_only":
        with st.spinner("📡 Fetching USA radar & forecast imagery..."):
            radar_frames = build_hrrr_assets()
        if not radar_frames:
            st.error("Failed to connect to WMS servers.")
            return
                
    elif mode == "rainviewer":
        with st.spinner("🌍 Fetching global current radar..."):
            radar_frames = build_rainviewer_frames()
        if not radar_frames:
            st.error("Failed to fetch RainViewer data.")
            return

    elif mode == "gfs":
        with st.spinner("🌍 Checking GFS data status..."):
            radar_frames = build_gfs_frames()
        if not radar_frames:
            st.error("Failed to process GFS data. Ensure `eccodes` and `cfgrib` are installed correctly.")
            return

    if radar_frames:
        if mode == "gfs":
            with st.spinner("🌡️ Fetching temperature grid..."):
                target_dts = [f["dt"] for f in radar_frames]
                # We need the full global bounding box for the GFS map
                t_lat_min, t_lat_max, t_lon_min, t_lon_max = -85.0, 85.0, -180.0, 180.0
                    
                temp_grids = fetch_and_build_temp_grid(target_dts, t_lat_min, t_lat_max, t_lon_min, t_lon_max)
                for i, frame in enumerate(radar_frames):
                    frame["grid"] = temp_grids[i] if i < len(temp_grids) else []
        else:
            for frame in radar_frames:
                frame["grid"] = []

    # --- Build JS payload ---
    js_frames_array = ",\n".join(
        [f"{{ ts: {int(f['dt'].timestamp())}, time: '{f['time']}', img: '{f['img']}', rv_path: '{f.get('rv_path', '')}', grid: {json.dumps(f.get('grid', []))} }}" for f in radar_frames]
    )

    play_display = 'none' if len(radar_frames) <= 1 else 'flex'
    counter_display = 'none' if len(radar_frames) <= 1 else 'block'
    temp_display = 'flex' if mode == 'gfs' else 'none'

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
        <style>
            body {{ margin: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                   display: flex; flex-direction: column; align-items: center; padding: 2px 0; transition: background 0.3s; }}
            #app-container {{ display: flex; flex-direction: column; gap: 12px; align-items: center; width: 100%; max-width: {WIDTH}px; }}
            #map-container {{ position: relative; width: 100%; height: {HEIGHT}px; border-radius: 12px;
                            border: 1px solid #cbd5e1; box-shadow: 0 4px 12px rgba(0,0,0,0.1); overflow: hidden; }}
            body.dark-theme #map-container {{ border-color: #334155; }}
            #map {{ width: 100%; height: 100%; background: #cce4f0; }}
            body.dark-theme #map {{ background: #1e293b; }}
            .radar-blend {{ mix-blend-mode: multiply; }}
            body.dark-theme .radar-blend {{ mix-blend-mode: screen; }}
            
            .temp-point {{ width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
                          font-family: ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 11px; color: #0f172a;
                          box-shadow: 0 3px 6px rgba(0,0,0,0.4), inset 0 2px 4px rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.15);
                          margin-left: -12px; margin-top: -12px; }}
            
            #controls-wrapper {{ background: #ffffff; color: #1e293b; padding: 10px 20px; border-radius: 12px; width: 100%; box-sizing: border-box;
                                 box-shadow: 0 2px 8px rgba(0,0,0,0.05); display: flex; flex-direction: row; align-items: center; gap: 15px;
                                 border: 1px solid #e2e8f0; transition: all 0.3s; flex-wrap: wrap; justify-content: center; }}
            body.dark-theme #controls-wrapper {{ background: #1e293b; color: #f8fafc; border-color: #334155; }}
            
            #playBtn {{ background: #2563eb; border: none; color: white; width: 32px; height: 32px; border-radius: 50%; cursor: pointer;
                       font-size: 12px; display: {play_display}; align-items: center; justify-content: center; flex-shrink: 0; }}
            #playBtn:hover {{ background: #1d4ed8; }}
            
            .slider-container {{ flex-grow: 1; display: {play_display}; align-items: center; min-width: 200px; }}
            input[type="range"] {{ -webkit-appearance: none; width: 100%; background: transparent; cursor: pointer; }}
            input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%; background: #2563eb; margin-top: -5px; }}
            input[type="range"]::-webkit-slider-runnable-track {{ width: 100%; height: 6px; background: #cbd5e1; border-radius: 3px; }}
            body.dark-theme input[type="range"]::-webkit-slider-runnable-track {{ background: #475569; }}
            
            #time-display {{ font-size: 16px; font-weight: 800; color: #0f172a; min-width: 380px; text-align: center; white-space: nowrap; }}
            body.dark-theme #time-display {{ color: #f8fafc; }}
            
            .toggle-group {{ display: flex; background: #f1f5f9; border-radius: 8px; padding: 4px; gap: 4px; border: 1px solid #e2e8f0; }}
            body.dark-theme .toggle-group {{ background: #0f172a; border-color: #334155; }}
            .toggle-group label {{ font-size: 12px; font-weight: 700; color: #64748b; padding: 4px 10px; border-radius: 6px; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; white-space: nowrap; }}
            body.dark-theme .toggle-group label {{ color: #94a3b8; }}
            .toggle-group input[type="radio"], .toggle-group input[type="checkbox"] {{ display: none; }}
            .toggle-group input[type="radio"]:checked + label, .toggle-group input[type="checkbox"]:checked + label {{ background: #ffffff; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            body.dark-theme .toggle-group input[type="radio"]:checked + label, body.dark-theme .toggle-group input[type="checkbox"]:checked + label {{ background: #334155; color: #f8fafc; }}
            
            #frame-counter {{ font-size: 12px; font-weight: 600; color: #64748b; min-width: 50px; text-align: center; display: {counter_display}; }}
            
            #loading-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.9); display: flex; align-items: center; justify-content: center; z-index: 9999; font-size: 16px; color: #64748b; border-radius: 12px; }}
            body.dark-theme #loading-overlay {{ background: rgba(30,41,59,0.9); color: #94a3b8; }}
            #loading-overlay.hidden {{ opacity: 0; pointer-events: none; }}
        </style>
    </head>
    <body>
        <div id="app-container">
            <div id="controls-wrapper">
                <button id="playBtn">&#9654;</button>
                <div class="slider-container">
                    <input type="range" id="slider" min="0" max="{len(radar_frames) - 1}" value="0">
                </div>
                <div id="frame-counter">1/{len(radar_frames)}</div>
                <div id="time-display">Loading...</div>
                
                <div class="toggle-group">
                    <input type="radio" id="t-light" name="mapTheme" value="light" checked>
                    <label for="t-light">☀️ Map</label>
                    <input type="radio" id="t-dark" name="mapTheme" value="dark">
                    <label for="t-dark">🌙 Dark</label>
                </div>
                
                <div class="toggle-group" style="display: {temp_display};">
                    <input type="checkbox" id="t-grid-toggle">
                    <label for="t-grid-toggle">🌡️ Temps</label>
                </div>
                
                <div class="toggle-group">
                    <input type="radio" id="f-raw" name="radarFilter" value="none">
                    <label for="f-raw">Raw</label>
                    <input type="radio" id="f-combined" name="radarFilter" value="combined" checked>
                    <label for="f-combined">3-Tier</label>
                    <input type="radio" id="f-cyan" name="radarFilter" value="cyan">
                    <label for="f-cyan">Light</label>
                    <input type="radio" id="f-blue" name="radarFilter" value="blue">
                    <label for="f-blue">Mod</label>
                    <input type="radio" id="f-red" name="radarFilter" value="red">
                    <label for="f-red">Heavy</label>
                </div>
            </div>
    
            <div id="map-container">
                <div id="map"></div>
                <div id="loading-overlay">Loading map data…</div>
            </div>
        </div>
    
        <script>
            const appMode = "{mode}";
            const frames = [{js_frames_array}];
            const totalFrames = frames.length;
            const activeBounds = {active_bounds};
            
            const map = L.map('map', {{ zoomControl: false, minZoom: 2, maxZoom: 10, zoomSnap: 0 }}).fitBounds(activeBounds);
            L.control.zoom({{ position: 'topright' }}).addTo(map);
    
            map.createPane('rainviewerPane');
            map.getPane('rainviewerPane').style.zIndex = 400;
            map.getPane('rainviewerPane').classList.add('radar-blend');
    
            map.createPane('primaryPane');
            map.getPane('primaryPane').style.zIndex = 410;
            if (appMode !== 'gfs') map.getPane('primaryPane').classList.add('radar-blend');
    
            const basemaps = {{
                light: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }}),
                dark: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }})
            }};
            basemaps.light.addTo(map);
            
            let primaryLayer = L.imageOverlay('', activeBounds, {{pane: 'primaryPane', opacity: 0, interactive: false}}).addTo(map);
            let rainviewerLayer = null;
            
            const slider = document.getElementById('slider');
            const timeDisplay = document.getElementById('time-display');
            const frameCounter = document.getElementById('frame-counter');
            const playBtn = document.getElementById('playBtn');
            const tGridToggle = document.getElementById('t-grid-toggle');
            const loadingOverlay = document.getElementById('loading-overlay');
    
            let timer = null;
            let isPlaying = false;
            let showTempGrid = tGridToggle.checked;
            let activeFilter = 'combined';
            let frameCache = {{}};
            let tempLayerGroup = L.layerGroup().addTo(map);
            let currentDrawIndex = -1; 
    
            function applyColorFilter(ctx, width, height) {{
                if (activeFilter === 'none') return;
                let imgData = ctx.getImageData(0, 0, width, height);
                let data = imgData.data;
                let pixelTiers = new Uint8Array(width * height);
    
                for (let i = 0; i < data.length; i += 4) {{
                    let pxIdx = i / 4;
                    let origAlpha = data[i+3];
                    if (origAlpha < 100) {{ data[i+3] = 0; pixelTiers[pxIdx] = 0; continue; }}
    
                    let r = data[i], g = data[i+1], b = data[i+2];
                    let isSevere = (r > 200 && g < 80);  
                    let isMod = (r > 200 && g >= 80 && g < 220 && b < 100); 
                    let outAlpha = origAlpha < 200 ? origAlpha + 55 : 255;
                    let currentTier = 0;
    
                    if (activeFilter === 'combined') {{
                        if (isSevere) {{ currentTier = 3; data[i]=255; data[i+1]=0; data[i+2]=0; }}
                        else if (isMod) {{ currentTier = 2; data[i]=0; data[i+1]=0; data[i+2]=255; }}
                        else {{ currentTier = 1; data[i]=0; data[i+1]=255; data[i+2]=255; }}
                    }} 
                    else if (activeFilter === 'cyan') {{ 
                        if (!isMod && !isSevere) {{ currentTier = 1; data[i]=0; data[i+1]=255; data[i+2]=255; }}
                        else {{ data[i+3] = 0; }}
                    }} 
                    else if (activeFilter === 'blue') {{
                        if (isMod && !isSevere) {{ currentTier = 2; data[i]=0; data[i+1]=0; data[i+2]=255; }}
                        else {{ data[i+3] = 0; }}
                    }} 
                    else if (activeFilter === 'red') {{ 
                        if (isSevere) {{ currentTier = 3; data[i]=255; data[i+1]=0; data[i+2]=0; }}
                        else {{ data[i+3] = 0; }}
                    }}
                    
                    if (currentTier > 0) data[i+3] = outAlpha;
                    pixelTiers[pxIdx] = currentTier;
                }}
    
                let grey = 160;
                for (let y = 1; y < height - 1; y++) {{
                    for (let x = 1; x < width - 1; x++) {{
                        let idx = y * width + x;
                        let myTier = pixelTiers[idx];
                        if (myTier === 1 || myTier === 2) {{
                            if (pixelTiers[idx-1] !== myTier || pixelTiers[idx+1] !== myTier ||
                                pixelTiers[idx-width] !== myTier || pixelTiers[idx+width] !== myTier) {{
                                let di = idx * 4;
                                data[di] = grey; data[di+1] = grey; data[di+2] = grey;
                            }}
                        }}
                    }}
                }}
                ctx.putImageData(imgData, 0, 0);
            }}
    
            function getFilteredImage(index, callback) {{
                if (activeFilter === 'none') {{ callback(frames[index].img); return; }}
                if (frameCache[activeFilter] && frameCache[activeFilter][index]) {{ callback(frameCache[activeFilter][index]); return; }}
                
                let canvas = document.createElement('canvas');
                let img = new Image();
                img.onload = () => {{
                    canvas.width = img.naturalWidth; 
                    canvas.height = img.naturalHeight;
                    let ctx = canvas.getContext('2d', {{willReadFrequently: true}});
                    ctx.drawImage(img, 0, 0);
                    applyColorFilter(ctx, canvas.width, canvas.height);
                    let dataUrl = canvas.toDataURL();
                    if (!frameCache[activeFilter]) frameCache[activeFilter] = {{}};
                    frameCache[activeFilter][index] = dataUrl;
                    callback(dataUrl);
                }};
                img.onerror = () => {{ callback(frames[index].img); }};
                img.src = frames[index].img;
            }}
    
            function setPrimaryOverlay(url) {{
                if (primaryLayer._url === url) {{ primaryLayer.setOpacity(0.85); return; }}
                primaryLayer.setOpacity(0);
                primaryLayer.off('load');
                primaryLayer.once('load', () => {{ primaryLayer.setOpacity(0.85); }});
                primaryLayer.setUrl(url);
                setTimeout(() => {{ if (primaryLayer._url === url && primaryLayer.options.opacity === 0) primaryLayer.setOpacity(0.85); }}, 60);
            }}
    
            function getTempColor(t) {{
                if (t < 10) return '#c4b5fd'; if (t < 25) return '#93c5fd'; if (t < 40) return '#67e8f9'; 
                if (t < 55) return '#86efac'; if (t < 70) return '#fde047'; if (t < 85) return '#fdba74'; 
                if (t < 95) return '#f87171'; return '#fca5a5'; 
            }}
    
            function drawFrame(index) {{
                if (!frames[index]) return;
                
                let frameIdx = parseInt(index);
                currentDrawIndex = frameIdx;
                let targetFrame = frames[frameIdx];
                let requestedFilter = activeFilter;
                
                if (targetFrame.img !== 'rv_native') {{
                    getFilteredImage(frameIdx, (url) => {{
                        if (currentDrawIndex !== frameIdx) return;
                        if (activeFilter !== requestedFilter) return; 
                        setPrimaryOverlay(url);
                    }});
                }} else {{
                    primaryLayer.setOpacity(0); 
                }}
                
                if (appMode === 'rainviewer' && rainviewerLayer) {{
                    let newUrlBase = targetFrame.rv_path;
                    if (rainviewerLayer.options.urlBase !== newUrlBase) {{
                        rainviewerLayer.options.urlBase = newUrlBase;
                        rainviewerLayer.redraw();
                    }}
                    if (!map.hasLayer(rainviewerLayer)) map.addLayer(rainviewerLayer);
                    rainviewerLayer.setOpacity(0.85);
                }}
    
                timeDisplay.innerText = targetFrame.time;
                frameCounter.innerText = (frameIdx + 1) + "/" + totalFrames;
                
                tempLayerGroup.clearLayers();
                if (showTempGrid && targetFrame.grid) {{
                    targetFrame.grid.forEach(pt => {{
                        let color = getTempColor(pt.t);
                        let icon = L.divIcon({{
                            className: 'custom-temp', iconSize: null,
                            html: `<div class="temp-point" style="background-color: ${{color}}E6;">${{pt.t}}</div>`
                        }});
                        L.marker([pt.lat, pt.lon], {{icon: icon, interactive: false}}).addTo(tempLayerGroup);
                    }});
                }}
            }}
    
            function nextFrame() {{ let n = parseInt(slider.value) + 1; if (n > totalFrames - 1) n = 0; slider.value = n; drawFrame(n); }}
            function prevFrame() {{ let n = parseInt(slider.value) - 1; if (n < 0) n = totalFrames - 1; slider.value = n; drawFrame(n); }}
    
            playBtn.onclick = () => {{
                if (isPlaying) {{ clearInterval(timer); playBtn.innerHTML = "&#9654;"; isPlaying = false; }}
                else {{ timer = setInterval(nextFrame, 400); playBtn.innerHTML = "&#10074;&#10074;"; isPlaying = true; }}
            }};
            
            slider.oninput = (e) => {{ if (isPlaying) playBtn.click(); drawFrame(e.target.value); }};
            tGridToggle.addEventListener('change', (e) => {{ showTempGrid = e.target.checked; drawFrame(slider.value); }});
    
            document.addEventListener('keydown', (e) => {{
                if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
                if (e.code === 'Space') {{ e.preventDefault(); playBtn.click(); }}
                else if (e.code === 'ArrowLeft') {{ e.preventDefault(); if (isPlaying) playBtn.click(); prevFrame(); }}
                else if (e.code === 'ArrowRight') {{ e.preventDefault(); if (isPlaying) playBtn.click(); nextFrame(); }}
            }});
    
            document.querySelectorAll('input[name="mapTheme"]').forEach(radio => {{
                radio.addEventListener('change', (e) => {{
                    if (e.target.value === 'dark') {{ document.body.classList.add('dark-theme'); map.removeLayer(basemaps.light); map.addLayer(basemaps.dark); }}
                    else {{ document.body.classList.remove('dark-theme'); map.removeLayer(basemaps.dark); map.addLayer(basemaps.light); }}
                }});
            }});
    
            document.querySelectorAll('input[name="radarFilter"]').forEach(radio => {{
                radio.addEventListener('change', (e) => {{
                    activeFilter = e.target.value;
                    if (rainviewerLayer) rainviewerLayer.redraw();
                    drawFrame(slider.value);
                }});
            }});
    
            L.RainviewerCanvasLayer = L.GridLayer.extend({{
                createTile: function(coords, done) {{
                    var tile = document.createElement('canvas');
                    tile.width = tile.height = 256;
                    var ctx = tile.getContext('2d', {{willReadFrequently: true}});
                    var img = new Image();
                    img.crossOrigin = 'Anonymous';
                    img.src = `${{this.options.urlBase}}/256/${{coords.z}}/${{coords.x}}/${{coords.y}}/2/1_1.png`;
                     
                    img.onload = function() {{
                        ctx.drawImage(img, 0, 0, 256, 256);
                        if (activeFilter !== 'none') applyColorFilter(ctx, 256, 256);
                        done(null, tile);
                    }};
                    img.onerror = function() {{ done(null, tile); }};
                    return tile;
                }}
            }});
    
            if (appMode === 'rainviewer') {{
                rainviewerLayer = new L.RainviewerCanvasLayer({{
                    urlBase: frames.length > 0 ? frames[0].rv_path : '',
                    pane: 'rainviewerPane',
                    maxNativeZoom: 9
                }});
            }}
    
            drawFrame(0);
            loadingOverlay.classList.add('hidden');
            
            if (totalFrames > 1) {{
                setTimeout(() => playBtn.click(), 600);
            }}
            
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=760)

# --- App Render ---
tab1, tab2 = st.tabs(["🇺🇸 USA High-Res", "🌍 Global Map"])

with tab1:
    render_flipbook("hrrr_only")

with tab2:
    col_radio, col_btn = st.columns([4, 1])
    with col_radio:
        global_mode = st.radio(
            "Global Data Source:", 
            ["Live Radar (RainViewer)", "16-Day Forecast (GFS)"], 
            horizontal=True, 
            label_visibility="collapsed"
        )
    
    with col_btn:
        if "Forecast" in global_mode and st.button("🗑️ Clear Cache", help="Clear local GFS data and Temp grids"):
            for file in os.listdir(GFS_DIR):
                file_path = os.path.join(GFS_DIR, file)
                if os.path.isfile(file_path): os.remove(file_path)
            st.cache_data.clear()
            st.rerun()

    if "RainViewer" in global_mode:
        render_flipbook("rainviewer")
    else:
        render_flipbook("gfs")
