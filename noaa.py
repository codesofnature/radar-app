/* --- YACHT ANIMATION CSS (3/4 Perspective) --- */
.yacht-container {{ pointer-events: none; }}
#yacht-icon {{
    width: 100%; height: 100%;
    transition: filter 2s ease-out;
    filter: drop-shadow(-2px 3px 3px rgba(0,0,0,0.5));
}}
.yacht-wake {{
    position: absolute;
    bottom: 0px; left: 50%;
    transform: translateX(-50%);
    width: 34px; height: 12px;
    background: radial-gradient(ellipse at center, rgba(255,255,255,0.75) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
    animation: wakePulse 1.8s ease-in-out infinite;
    pointer-events: none;
}}
@keyframes wakePulse {{
    0%, 100% {{ opacity: 0.5; transform: translateX(-50%) scale(0.8); }}
    50% {{ opacity: 0.9; transform: translateX(-50%) scale(1.2); }}
}}

/* --- B-2 BOMBER GROUND SHADOW --- */
#plane-shadow {{
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    height: 20%;
    background: radial-gradient(ellipse, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.2) 40%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    transition: bottom 4s ease-out, opacity 4s ease-out, width 4s ease-out;
    bottom: -4px;
    opacity: 0.85;
}}
