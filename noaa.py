import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import concurrent.futures
import math
import io
import os
import time
import logging
import numpy as np
from PIL import Image
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Instant Radar", layout="wide", page_icon="⚡")

# --- Configuration ---
LOCAL_TZ = ZoneInfo("America/New_York")

# Reduced resolution factor. 1.5 is plenty sharp for a car display 
# and drastically reduces download size and memory footprint.
BBOX = "-14200000,2700000,-7200000,6400000"
WIDTH = 1200
HEIGHT = 700
RADAR_RES_FACTOR = 1.5
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

def process_radar_image(img_bytes):
    """Applies the 3-Tier color filter on the server side using numpy."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        data = np.array(img)
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        out = np.zeros_like(data)
        
        # Identify pixels with enough opacity
        valid = a >= 100
        
        # Color tier logic
        is_severe = valid & (r > 200) & (g < 80)
        is_mod = valid & (r > 200) & (g >= 80) & (g < 220) & (b < 100)
        is_light = valid & ~is_severe & ~is_mod
        
        # Apply strict 3-tier colors
        out[is_severe, 0], out[is_severe, 1], out[is_severe, 2] = 255, 0, 0       # Red
        out[is_mod, 0], out[is_mod, 1], out[is_mod, 2] = 0, 0, 255                # Blue
        out[is_light, 0], out[is_light, 1], out[is_light, 2] = 0, 255, 255        # Cyan
        
        # Adjust alpha
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
        if not dt_str.endswith("Z"): dt_str += "Z"
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception as e:
        now = datetime.now(timezone.utc)
        return now.replace(minute=0, second=0, microsecond=0)

def fetch_single_image(url_info, max_retries=2):
    frame_time, time_str, url = url_info
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                # Process the image colors in Python
                processed_bytes = process_radar_image(resp.content)
                b64 = base64.b64encode(processed_bytes).decode("utf-8")
                return f"data:image/png;base64,{b64}"
            elif resp.status_code == 404:
                return None
        except Exception:
            if attempt < max_retries: time.sleep(0.5 * (attempt + 1))
    return None

@st.cache_data(ttl=300, show_spinner=False)
def build_hrrr_assets():
    init_time = get_model_init_time()
    now = datetime.now(timezone.utc)
    frames_data = []
    
    live_wms_url = f"https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0q.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=nexrad-n0q-900913&FORMAT=image/png&TRANSPARENT=true&WIDTH={RADAR_W}&HEIGHT={RADAR_H}&SRS=EPSG:3857&BBOX={BBOX}"
    live_label = now.astimezone(LOCAL_TZ).strftime("%a, %b %d - %I:%M %p (Live)")
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
                
                # Simplified date/time string
                time_str = local_time.strftime("%a, %b %d - %I:%M %p")
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
        
        init_time = init_time - timedelta(hours=1)

    return frames_data

# --- Flipbook Renderer ---
def render_flipbook():
    with st.spinner("📡 Processing radar imagery..."):
        radar_frames = build_hrrr_assets()
        
    if not radar_frames:
        st.error("Failed to fetch radar imagery.")
        return

    # Pass the pre-processed base64 images straight to Javascript
    js_frames_array = ",\n".join(
        [f"{{ time: '{f['time']}', img: '{f['img']}' }}" for f in radar_frames]
    )

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
        <style>
            body {{ margin: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                   display: flex; flex-direction: column; align-items: center; padding: 2px 0; }}
            #app-container {{ display: flex; flex-direction: column; gap: 12px; align-items: center; width: 100%; max-width: {WIDTH}px; }}
            #map-container {{ position: relative; width: 100%; height: {HEIGHT}px; border-radius: 12px;
                            border: 1px solid #cbd5e1; box-shadow: 0 4px 12px rgba(0,0,0,0.1); overflow: hidden; }}
            #map {{ width: 100%; height: 100%; background: #cce4f0; }}
            
            .radar-blend {{ mix-blend-mode: multiply; }}
            
            #controls-wrapper {{ background: #ffffff; color: #1e293b; padding: 12px 24px; border-radius: 12px; width: 100%; box-sizing: border-box;
                                 box-shadow: 0 2px 8px rgba(0,0,0,0.05); display: flex; flex-direction: row; align-items: center; gap: 20px;
                                 border: 1px solid #e2e8f0; justify-content: space-between; }}
            
            #playBtn {{ background: #2563eb; border: none; color: white; width: 44px; height: 44px; border-radius: 50%; cursor: pointer;
                       font-size: 16px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: background 0.2s; }}
            #playBtn:hover {{ background: #1d4ed8; }}
            
            .slider-container {{ flex-grow: 1; display: flex; align-items: center; padding: 0 10px; }}
            
            /* Enhanced Slider Styling */
            input[type="range"] {{ -webkit-appearance: none; width: 100%; background: transparent; cursor: pointer; margin: 0; height: 24px; }}
            input[type="range"]:focus {{ outline: none; }}
            input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance: none; height: 22px; width: 22px; border-radius: 50%; background: #2563eb; 
                                                         margin-top: -8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); border: 2px solid #fff; }}
            input[type="range"]::-webkit-slider-runnable-track {{ width: 100%; height: 6px; background: #cbd5e1; border-radius: 3px; }}
            
            #time-display {{ font-size: 18px; font-weight: 700; color: #0f172a; min-width: 280px; text-align: right; white-space: nowrap; letter-spacing: -0.5px; }}
            
            #loading-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.9); 
                                display: flex; align-items: center; justify-content: center; z-index: 9999; font-size: 16px; 
                                color: #64748b; border-radius: 12px; transition: opacity 0.3s; }}
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
                <div id="time-display">Loading...</div>
            </div>
    
            <div id="map-container">
                <div id="map"></div>
                <div id="loading-overlay">Initializing Map…</div>
            </div>
        </div>
    
        <script>
            const frames = [{js_frames_array}];
            const totalFrames = frames.length;
            const activeBounds = {MAP_BOUNDS};
            
            const map = L.map('map', {{ 
                zoomControl: false, 
                minZoom: 4, 
                maxZoom: 10, 
                zoomSnap: 0,
                maxBounds: activeBounds,
                maxBoundsViscosity: 1.0 
            }}).fitBounds(activeBounds);
            
            L.control.zoom({{ position: 'topright' }}).addTo(map);
    
            map.createPane('primaryPane');
            map.getPane('primaryPane').style.zIndex = 410;
            map.getPane('primaryPane').classList.add('radar-blend');
    
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CARTO' }}).addTo(map);
            
            let primaryLayer = L.imageOverlay('', activeBounds, {{pane: 'primaryPane', opacity: 0.85, interactive: false}}).addTo(map);
            
            const slider = document.getElementById('slider');
            const timeDisplay = document.getElementById('time-display');
            const playBtn = document.getElementById('playBtn');
            const loadingOverlay = document.getElementById('loading-overlay');
    
            let timer = null;
            let isPlaying = false;
    
            function drawFrame(index) {{
                if (!frames[index]) return;
                
                // No canvas processing needed, just set the pre-processed URL directly
                let targetFrame = frames[index];
                primaryLayer.setUrl(targetFrame.img);
                timeDisplay.innerText = targetFrame.time;
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
                if (isPlaying) {{ 
                    clearInterval(timer); 
                    playBtn.innerHTML = "&#9654;"; // Play icon
                    isPlaying = false; 
                }} else {{ 
                    timer = setInterval(nextFrame, 450); // Smoother playback interval
                    playBtn.innerHTML = "&#10074;&#10074;"; // Pause icon
                    isPlaying = true; 
                }}
            }};
            
            slider.oninput = (e) => {{ 
                if (isPlaying) playBtn.click(); // Auto-pause if user grabs slider
                drawFrame(e.target.value); 
            }};
    
            document.addEventListener('keydown', (e) => {{
                if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
                if (e.code === 'Space') {{ e.preventDefault(); playBtn.click(); }}
                else if (e.code === 'ArrowLeft') {{ e.preventDefault(); if (isPlaying) playBtn.click(); prevFrame(); }}
                else if (e.code === 'ArrowRight') {{ e.preventDefault(); if (isPlaying) playBtn.click(); nextFrame(); }}
            }});
    
            // Initialize first frame
            drawFrame(0);
            loadingOverlay.classList.add('hidden');
            
            // Auto-play on load
            if (totalFrames > 1) {{
                setTimeout(() => playBtn.click(), 500);
            }}
            
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=760)

# --- App Render ---
render_flipbook()
