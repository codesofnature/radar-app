import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import concurrent.futures
import math
import json
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- Page Config ---
st.set_page_config(page_title="Instant Radar Flipbook", layout="wide", page_icon="⚡")

# --- Configuration ---
# Set your target local timezone here (Handles DST automatically)
LOCAL_TZ = ZoneInfo("America/New_York")

# --- STATIC SERVING WORKAROUND (Fixes Base64 Bloat) ---
# To use this: set USE_STATIC_SERVING = True 
# AND run your app with: streamlit run noaa.py --server.enableStaticServing=true
USE_STATIC_SERVING = False 
STATIC_DIR = "static/radar_frames"
if USE_STATIC_SERVING:
    os.makedirs(STATIC_DIR, exist_ok=True)

BBOX = "-14200000,2700000,-7200000,6400000"
WIDTH = 1200
HEIGHT = 650

RADAR_RES_FACTOR = 3
RADAR_W = int(WIDTH * RADAR_RES_FACTOR)
RADAR_H = int(HEIGHT * RADAR_RES_FACTOR)

MINUTES_OFFSETS = list(range(0, 18 * 60, 15)) + list(range(18 * 60, 49 * 60, 60))
TOTAL_FRAMES = len(MINUTES_OFFSETS)

def mercator_to_latlon(x, y):
    r = 6378137.0
    lon = math.degrees(x / r)
    lat = math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2.0)
    return lat, lon

# Calculate Map Bounds for Leaflet
xmin, ymin, xmax, ymax = map(float, BBOX.split(","))
lat_min, lon_min = mercator_to_latlon(xmin, ymin)
lat_max, lon_max = mercator_to_latlon(xmax, ymax)
MAP_BOUNDS = f"[[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]]"

@st.cache_data(ttl=300) 
def get_model_init_time():
    try:
        res = requests.get('https://mesonet.agron.iastate.edu/data/gis/images/4326/hrrr/refd_1080.json', timeout=5)
        dt_str = res.json()['model_init_utc']
        if not dt_str.endswith('Z'): dt_str += 'Z'
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"Error fetching model init time: {e}")
        now = datetime.now(timezone.utc)
        return now.replace(minute=0, second=0, microsecond=0)

def fetch_single_image(u_info):
    frame_time, time_str, url = u_info
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and 'image' in resp.headers.get('Content-Type', ''):
            if USE_STATIC_SERVING:
                filename = f"frame_{int(frame_time.timestamp())}.png"
                filepath = os.path.join(STATIC_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                # Streamlit statically serves the 'static' folder at '/app/static/'
                return f"app/static/radar_frames/{filename}"
            else:
                b64_str = base64.b64encode(resp.content).decode('utf-8')
                return f"data:image/png;base64,{b64_str}"
    except Exception as e:
        print(f"Error fetching WMS image from {url}: {e}")
    return None

@st.cache_data(ttl=300) 
def build_flipbook_assets():
    init_time = get_model_init_time()
    now = datetime.now(timezone.utc)
    
    frames_data = []
    
    live_wms_url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0q-900913&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&SRS=EPSG:3857&BBOX={BBOX}"
    
    live_info = (now, now.astimezone(LOCAL_TZ).strftime("🔴 USA NEXRAD: %A, %I:%M %p"), live_wms_url)
    live_img_src = fetch_single_image(live_info)
    
    if live_img_src:
        frames_data.append({
            "dt": now, 
            "time": live_info[1], 
            "img": live_img_src
        })

    for attempt in range(3):
        hrrr_frames = []
        urls_to_fetch = []
        
        for mins_offset in MINUTES_OFFSETS:
            frame_time = init_time + timedelta(minutes=mins_offset)
            if frame_time > now:
                layer_name = f"refd_{str(mins_offset).zfill(4)}"
                url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/hrrr/refd.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer_name}&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&CRS=EPSG:3857&BBOX={BBOX}"
                local_frame_time = frame_time.astimezone(LOCAL_TZ)
                time_str = local_frame_time.strftime("🔮 USA HRRR: %A, %I:%M %p")
                urls_to_fetch.append((frame_time, time_str, url))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(fetch_single_image, u): u for u in urls_to_fetch}
            for future in concurrent.futures.as_completed(future_to_url):
                u_info = future_to_url[future]
                img_src = future.result()
                if img_src:
                    hrrr_frames.append({"dt": u_info[0], "time": u_info[1], "img": img_src})
        
        if len(hrrr_frames) > 5: 
            hrrr_frames.sort(key=lambda x: x["dt"])
            frames_data.extend(hrrr_frames)
            return frames_data
            
        print(f"Model run for {init_time} incomplete. Falling back to previous hour...")
        init_time = init_time - timedelta(hours=1)
        
    return frames_data

@st.cache_data(ttl=1800)
def fetch_and_build_temp_grid(target_dts):
    points = []
    cols, rows = 36, 24 
    
    for r in range(rows):
        lat = 80 - (140 * r / (rows - 1))
        for c in range(cols):
            lon = -180 + (360 * c / cols)
            points.append({"lat": round(lat, 2), "lon": round(lon, 2)})
            
    chunk_size = 90
    all_hourly_data = []
    
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        lats = ",".join([str(p['lat']) for p in chunk])
        lons = ",".join([str(p['lon']) for p in chunk])
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&hourly=temperature_2m&temperature_unit=fahrenheit&past_days=1&forecast_days=2"
        
        try:
            data = requests.get(url, timeout=15).json()
            if not isinstance(data, list): data = [data]
            all_hourly_data.extend(data)
        except Exception as e:
            print(f"Error fetching Open-Meteo chunk {i}: {e}")
            all_hourly_data.extend([{} for _ in chunk])
            
        # Pacing delay to avoid triggering Open-Meteo API free-tier limits
        time.sleep(0.25)

    frames_grids = []
    for target_dt in target_dts:
        target_ts = target_dt.timestamp()
        grid = []
        for i, point in enumerate(points):
            try:
                hourly = all_hourly_data[i].get('hourly', None)
                if not hourly: continue
                
                times_ts = [datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp() for t in hourly['time']]
                temps = hourly['temperature_2m']
                
                temp_val = None
                if target_ts <= times_ts[0]: temp_val = temps[0]
                elif target_ts >= times_ts[-1]: temp_val = temps[-1]
                else:
                    for j in range(len(times_ts) - 1):
                        if times_ts[j] <= target_ts <= times_ts[j+1]:
                            t0, t1 = times_ts[j], times_ts[j+1]
                            v0, v1 = temps[j], temps[j+1]
                            if v0 is not None and v1 is not None:
                                temp_val = v0 + (v1 - v0) * ((target_ts - t0) / (t1 - t0))
                            break
                if temp_val is not None:
                    grid.append({"lat": point['lat'], "lon": point['lon'], "t": int(round(temp_val))})
            except Exception:
                pass
        frames_grids.append(grid)
    return frames_grids

with st.spinner("Fetching Satellite & Radar Geometry..."):
    radar_frames = build_flipbook_assets()
    if radar_frames:
        target_dts = [f["dt"] for f in radar_frames]
        temp_grids = fetch_and_build_temp_grid(target_dts)
        for i, frame in enumerate(radar_frames):
            frame["grid"] = temp_grids[i] if temp_grids else []

if not radar_frames:
    st.error("Failed to connect to WMS servers after multiple fallbacks. NOAA servers may be down.")
    st.stop()

# Build the JS Array Payload. Notice `f['img']` injects either Base64 or the static URL natively now.
js_frames_array = ",\n".join([f"{{ ts: {int(f['dt'].timestamp())}, time: '{f['time']}', img: '{f['img']}', grid: {json.dumps(f.get('grid', []))} }}" for f in radar_frames])

html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <style>
        body {{ margin: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px 0; transition: background 0.3s; }}
        
        #app-container {{ display: flex; flex-direction: column; gap: 12px; align-items: center; width: 100%; max-width: {WIDTH}px; }}

        #map-container {{ position: relative; width: 100%; height: {HEIGHT}px; border-radius: 12px; border: 1px solid #cbd5e1; box-shadow: 0 4px 12px rgba(0,0,0,0.1); overflow: hidden; }}
        body.dark-theme #map-container {{ border-color: #334155; }}
        
        #map {{ width: 100%; height: 100%; background: #cce4f0; }}
        body.dark-theme #map {{ background: #1e293b; }}
        
        .radar-blend {{ mix-blend-mode: multiply; }} 
        body.dark-theme .radar-blend {{ mix-blend-mode: screen; }} 
        
        /* NEW STYLES: Shaded Colored Circles for Temperatures */
        .temp-point {{ 
            width: 24px; 
            height: 24px; 
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-weight: 700; 
            font-size: 11px; 
            letter-spacing: -0.5px;
            color: #0f172a; 
            box-shadow: 0 3px 6px rgba(0,0,0,0.4), inset 0 2px 4px rgba(255,255,255,0.6); 
            border: 1px solid rgba(0,0,0,0.15);
            margin-left: -12px; 
            margin-top: -12px; 
        }}
        
        #controls-wrapper {{ background: #ffffff; color: #1e293b; padding: 10px 20px; border-radius: 12px; width: 100%; box-sizing: border-box; box-shadow: 0 2px 8px rgba(0,0,0,0.05); display: flex; flex-direction: row; align-items: center; gap: 15px; border: 1px solid #e2e8f0; transition: all 0.3s; flex-wrap: wrap; justify-content: center;}}
        body.dark-theme #controls-wrapper {{ background: #1e293b; color: #f8fafc; border-color: #334155; }}
        
        #playBtn {{ background: #2563eb; border: none; color: white; width: 32px; height: 32px; border-radius: 50%; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center; padding: 0; transition: background 0.15s; flex-shrink: 0; }}
        #playBtn:hover {{ background: #1d4ed8; }}
        
        .slider-container {{ flex-grow: 1; display: flex; align-items: center; min-width: 200px; }}
        input[type="range"] {{ -webkit-appearance: none; width: 100%; background: transparent; cursor: pointer; }}
        input[type="range"]:focus {{ outline: none; }}
        input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%; background: #2563eb; margin-top: -5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
        input[type="range"]::-webkit-slider-runnable-track {{ width: 100%; height: 6px; background: #cbd5e1; border-radius: 3px; }}
        body.dark-theme input[type="range"]::-webkit-slider-runnable-track {{ background: #475569; }}

        #time-display {{ font-size: 16px; font-weight: 800; color: #0f172a; min-width: 380px; text-align: center; white-space: nowrap; }}
        body.dark-theme #time-display {{ color: #f8fafc; }}
        
        .toggle-group {{ display: flex; background: #f1f5f9; border-radius: 8px; padding: 4px; gap: 4px; border: 1px solid #e2e8f0; }}
        body.dark-theme .toggle-group {{ background: #0f172a; border-color: #334155; }}
        .toggle-group label {{ font-size: 12px; font-weight: 700; color: #64748b; padding: 4px 10px; border-radius: 6px; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; white-space: nowrap;}}
        body.dark-theme .toggle-group label {{ color: #94a3b8; }}
        .toggle-group input[type="radio"], .toggle-group input[type="checkbox"] {{ display: none; }}
        .toggle-group input[type="radio"]:checked + label, .toggle-group input[type="checkbox"]:checked + label {{ background: #ffffff; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        body.dark-theme .toggle-group input[type="radio"]:checked + label, body.dark-theme .toggle-group input[type="checkbox"]:checked + label {{ background: #334155; color: #f8fafc; }}
    </style>
</head>
<body>

    <div id="app-container">
        <div id="controls-wrapper">
            <button id="playBtn">&#9654;</button>
            <div class="slider-container">
                <input type="range" id="slider" min="0" max="{len(radar_frames) - 1}" value="0">
            </div>
            <div id="time-display">Loading...</div>

            <div class="toggle-group">
                <input type="radio" id="t-light" name="mapTheme" value="light" checked>
                <label for="t-light">☀️ Map</label>
                <input type="radio" id="t-dark" name="mapTheme" value="dark">
                <label for="t-dark">🌙 Dark</label>
            </div>
            
            <div class="toggle-group">
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
        </div>
    </div>

    <script>
        const frames = [
            {js_frames_array}
        ];
        
        let rainviewerFrames = [];
        const hrrrBounds = {MAP_BOUNDS};
        
        // --- Leaflet Initialization ---
        const map = L.map('map', {{
            zoomControl: false,
            minZoom: 3,
            maxZoom: 10,
            zoomSnap: 0 
        }}).fitBounds(hrrrBounds);
        L.control.zoom({{ position: 'topright' }}).addTo(map);

        map.createPane('rainviewerPane');
        map.getPane('rainviewerPane').style.zIndex = 400;
        map.getPane('rainviewerPane').classList.add('radar-blend');

        map.createPane('hrrrPane');
        map.getPane('hrrrPane').style.zIndex = 410;
        map.getPane('hrrrPane').classList.add('radar-blend');
        
        const basemaps = {{
            light: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }}),
            dark: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }})
        }};
        basemaps.light.addTo(map);

        let hrrrLayers = [
            L.imageOverlay('', hrrrBounds, {{pane: 'hrrrPane', opacity: 0.85, interactive: false}}).addTo(map),
            L.imageOverlay('', hrrrBounds, {{pane: 'hrrrPane', opacity: 0, interactive: false}}).addTo(map)
        ];
        let hrrrActiveBuffer = 0;

        let tempLayerGroup = L.layerGroup().addTo(map);
        let rainviewerLayers = {{}};
        
        const slider = document.getElementById('slider');
        const timeDisplay = document.getElementById('time-display');
        const playBtn = document.getElementById('playBtn');
        const tGridToggle = document.getElementById('t-grid-toggle');
        
        let timer = null;
        let isPlaying = false;
        let showTempGrid = tGridToggle.checked;
        let activeFilter = 'combined';
        let frameCache = {{}};
        let lastDrawnRvIdx = -1;

        function precalculateHrrrCache() {{
            frames.forEach((f, idx) => {{
                getFilteredHrrrImage(idx, () => {{}});
            }});
        }}

        fetch('https://api.rainviewer.com/public/weather-maps.json')
            .then(res => res.json())
            .then(data => {{
                rainviewerFrames = [...data.radar.past, ...data.radar.nowcast].map(f => ({{
                    urlBase: data.host + f.path,
                    time: f.time
                }}));
                
                rainviewerFrames.forEach((frame, idx) => {{
                    rainviewerLayers[idx] = new L.RainviewerCanvasLayer({{
                        urlBase: frame.urlBase,
                        pane: 'rainviewerPane',
                        maxNativeZoom: 9
                    }}).addTo(map);
                    rainviewerLayers[idx].setOpacity(0);
                }});

                precalculateHrrrCache();
                drawFrame(slider.value); 
            }})
            .catch(e => console.log("Rainviewer fetch failed", e));
        
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
                let isSevere = (r > 150 && g < 100); 
                let isMod = (r > 150 && g > 120 && b < 100);
                
                let currentTier = 0;
                let outAlpha = origAlpha < 200 ? origAlpha + 55 : 255; 

                if (activeFilter === 'combined') {{
                    if (isSevere) {{ currentTier = 3; data[i] = 255; data[i+1] = 0; data[i+2] = 0; data[i+3] = outAlpha; }}
                    else if (isMod) {{ currentTier = 2; data[i] = 0; data[i+1] = 0; data[i+2] = 255; data[i+3] = outAlpha; }}
                    else {{ currentTier = 1; data[i] = 0; data[i+1] = 255; data[i+2] = 255; data[i+3] = outAlpha; }}
                }} 
                else if (activeFilter === 'cyan') {{ currentTier = 1; data[i] = 0; data[i+1] = 255; data[i+2] = 255; data[i+3] = outAlpha; }}
                else if (activeFilter === 'blue') {{
                    if (isMod || isSevere) {{ currentTier = 2; data[i] = 0; data[i+1] = 0; data[i+2] = 255; data[i+3] = outAlpha; }}
                    else {{ data[i+3] = 0; }}
                }}
                else if (activeFilter === 'red') {{
                    if (isSevere) {{ currentTier = 3; data[i] = 255; data[i+1] = 0; data[i+2] = 0; data[i+3] = outAlpha; }}
                    else {{ data[i+3] = 0; }}
                }}
                pixelTiers[pxIdx] = currentTier;
            }}
            
            let greyValue = 160; 
            for (let y = 1; y < height - 1; y++) {{
                for (let x = 1; x < width - 1; x++) {{
                    let idx = y * width + x;
                    let myTier = pixelTiers[idx];
                    if (myTier === 1 || myTier === 2) {{
                        if (pixelTiers[idx - 1] !== myTier || pixelTiers[idx + 1] !== myTier ||
                            pixelTiers[idx - width] !== myTier || pixelTiers[idx + width] !== myTier) {{
                            let dataIdx = idx * 4;
                            data[dataIdx] = greyValue; data[dataIdx+1] = greyValue; data[dataIdx+2] = greyValue;
                        }}
                    }}
                }}
            }}
            ctx.putImageData(imgData, 0, 0);
        }}

        function getFilteredHrrrImage(index, callback) {{
            if (activeFilter === 'none') {{ callback(frames[index].img); return; }}
            if (frameCache[activeFilter] && frameCache[activeFilter][index]) {{ callback(frameCache[activeFilter][index]); return; }}
            
            let canvas = document.createElement('canvas');
            canvas.width = {RADAR_W}; canvas.height = {RADAR_H};
            let ctx = canvas.getContext('2d', {{willReadFrequently: true}});
            let img = new Image();
            img.onload = () => {{
                ctx.drawImage(img, 0, 0);
                applyColorFilter(ctx, canvas.width, canvas.height);
                let dataUrl = canvas.toDataURL();
                if (!frameCache[activeFilter]) frameCache[activeFilter] = {{}};
                frameCache[activeFilter][index] = dataUrl;
                callback(dataUrl);
            }};
            img.src = frames[index].img;
        }}

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
                    let nwLatLng = L.latLng(hrrrBounds[1][0], hrrrBounds[0][1]); 
                    let seLatLng = L.latLng(hrrrBounds[0][0], hrrrBounds[1][1]); 
                    let pNw = map.project(nwLatLng, coords.z);
                    let pSe = map.project(seLatLng, coords.z);
                    let tileNwX = coords.x * 256;
                    let tileNwY = coords.y * 256;
                    let worldTiles = Math.pow(2, coords.z);
                    let wrapOffset = Math.floor(coords.x / worldTiles) * (worldTiles * 256);
                    let maskX = Math.round((pNw.x + wrapOffset) - tileNwX);
                    let maskY = Math.round(pNw.y - tileNwY);
                    let maskW = Math.round(pSe.x - pNw.x);
                    let maskH = Math.round(pSe.y - pNw.y);
                    
                    if (maskX < 256 && maskY < 256 && (maskX + maskW) > 0 && (maskY + maskH) > 0) {{
                        ctx.clearRect(maskX, maskY, maskW, maskH);
                    }}
                    if (activeFilter !== 'none') applyColorFilter(ctx, 256, 256);
                    done(null, tile);
                }};
                img.onerror = function(e) {{ done(null, tile); }};
                return tile;
            }}
        }});

        function setHrrrOverlay(url) {{
            let nextBuffer = 1 - hrrrActiveBuffer;
            let currentBuffer = hrrrActiveBuffer;
            
            hrrrLayers[nextBuffer].once('load', () => {{
                hrrrLayers[nextBuffer].setOpacity(0.85);
                hrrrLayers[currentBuffer].setOpacity(0);
            }});
            hrrrLayers[nextBuffer].setUrl(url);
            hrrrActiveBuffer = nextBuffer;
        }}

        function drawGlobalFrame(targetIdx) {{
            if (targetIdx === lastDrawnRvIdx) return;
            
            for (const [key, layer] of Object.entries(rainviewerLayers)) {{
                let k = parseInt(key);
                if (k === targetIdx) {{
                    layer.setOpacity(0.85);
                    if (layer.setZIndex) layer.setZIndex(10); 
                }} else if (k === lastDrawnRvIdx) {{
                    layer.setOpacity(0.85); 
                    if (layer.setZIndex) layer.setZIndex(5); 
                }} else {{
                    layer.setOpacity(0);
                    if (layer.setZIndex) layer.setZIndex(1);
                }}
            }}
            lastDrawnRvIdx = targetIdx;
        }}

        function getTempColor(t) {{
            if (t < 10) return '#c4b5fd'; if (t < 25) return '#93c5fd'; 
            if (t < 40) return '#67e8f9'; if (t < 55) return '#86efac'; 
            if (t < 70) return '#fde047'; if (t < 85) return '#fdba74'; 
            if (t < 95) return '#f87171'; return '#fca5a5'; 
        }}

        function drawFrame(index) {{
            if (!frames[index]) return;
            let targetTs = frames[index].ts;
            
            getFilteredHrrrImage(index, (url) => {{
                setHrrrOverlay(url);
            }});
            
            let globalStatus = "";
            if (rainviewerFrames.length > 0) {{
                let latestRvIdx = rainviewerFrames.length - 1;
                let latestRvTime = rainviewerFrames[latestRvIdx].time;
                let renderRvIdx = 0;
                
                if (targetTs <= latestRvTime + 3600) {{
                    let closestIdx = -1;
                    let minDiff = Infinity;
                    rainviewerFrames.forEach((rv, i) => {{
                        let diff = Math.abs(rv.time - targetTs);
                        if (diff < minDiff) {{ minDiff = diff; closestIdx = i; }}
                    }});
                    renderRvIdx = closestIdx !== -1 ? closestIdx : 0;
                    globalStatus = " 🌍 (Global Sync)";
                }} else {{
                    renderRvIdx = index % rainviewerFrames.length;
                    globalStatus = " 🌍 (Global Looping)";
                }}
                
                drawGlobalFrame(renderRvIdx);
            }}

            timeDisplay.innerText = frames[index].time + globalStatus;
            
            tempLayerGroup.clearLayers();
            if (showTempGrid && frames[index].grid) {{
                frames[index].grid.forEach(pt => {{
                    let color = getTempColor(pt.t);
                    let icon = L.divIcon({{
                        className: 'custom-temp',
                        iconSize: null,
                        /* NEW LOGIC: Dynamic background color with 90% opacity (E6) */
                        html: `<div class="temp-point" style="background-color: ${{color}}E6;">${{pt.t}}</div>`
                    }});
                    L.marker([pt.lat, pt.lon], {{icon: icon, interactive: false}}).addTo(tempLayerGroup);
                }});
            }}
        }}

        function nextFrame() {{
            let nextIdx = parseInt(slider.value) + 1;
            if (nextIdx > frames.length - 1) nextIdx = 0;
            slider.value = nextIdx;
            drawFrame(nextIdx);
        }}
        
        playBtn.onclick = () => {{
            if (isPlaying) {{ clearInterval(timer); playBtn.innerHTML = "&#9654;"; isPlaying = false; }} 
            else {{ timer = setInterval(nextFrame, 150); playBtn.innerHTML = "&#10074;&#10074;"; isPlaying = true; }}
        }};
        
        slider.oninput = (e) => {{
            if (isPlaying) playBtn.click();
            drawFrame(e.target.value);
        }};

        tGridToggle.addEventListener('change', (e) => {{
            showTempGrid = e.target.checked;
            drawFrame(slider.value);
        }});

        document.querySelectorAll('input[name="mapTheme"]').forEach(radio => {{
            radio.addEventListener('change', (e) => {{
                const mode = e.target.value;
                if (mode === 'dark') {{
                    document.body.classList.add('dark-theme');
                    map.removeLayer(basemaps.light);
                    map.addLayer(basemaps.dark);
                }} else {{
                    document.body.classList.remove('dark-theme');
                    map.removeLayer(basemaps.dark);
                    map.addLayer(basemaps.light);
                }}
            }});
        }});

        document.querySelectorAll('input[name="radarFilter"]').forEach(radio => {{
            radio.addEventListener('change', (e) => {{
                activeFilter = e.target.value;
                Object.values(rainviewerLayers).forEach(layer => layer.redraw());
                precalculateHrrrCache();
                drawFrame(slider.value);
            }});
        }});
        
        setTimeout(() => {{ playBtn.click(); }}, 800); 
    </script>
</body>
</html>
"""

components.html(html_code, height=750)
