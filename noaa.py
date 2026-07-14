// --- JELLY SWITCH: 3D SHADING & HEAD/TAIL PHYSICS ---
        let targetY = 0.0;      
        
        // Split the physics into a Head (fast) and a Tail (drags behind)
        let headY = 0.0;
        let tailY = 0.0;
        let headVel = 0;
        let tailVel = 0;

        function getUvY(element) {{
            const container = document.querySelector('.switch-container');
            const rect = element.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            const pixelY = (rect.top - containerRect.top) + (rect.height / 2);
            
            const h = container.clientHeight;
            const w = container.clientWidth;  
            if (jMaterial) jMaterial.uniforms.u_aspect.value = h / w;

            const vUvY = 1.0 - (pixelY / h); 
            return (vUvY - 0.5) * (h / w);
        }}

        function setTarget(index, mode, btnElement) {{
            targetY = getUvY(btnElement);
            document.querySelectorAll('.icon-btn').forEach(btn => btn.classList.remove('active'));
            btnElement.classList.add('active');
            
            if (typeof setLayerMode === 'function') setLayerMode(mode); 
        }}

        const jCanvas = document.getElementById('jelly-canvas');
        const jRenderer = new THREE.WebGLRenderer({{ canvas: jCanvas, alpha: true, antialias: true }});
        jRenderer.setPixelRatio(window.devicePixelRatio || 1);
        jRenderer.setSize(64, 180);

        const jScene = new THREE.Scene();
        const jCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        let jMaterial = new THREE.ShaderMaterial({{
            uniforms: {{
                u_headY: {{ value: headY }},
                u_tailY: {{ value: tailY }},
                u_aspect: {{ value: 180.0 / 64.0 }}
            }},
            vertexShader: `
                varying vec2 vUv;
                void main() {{
                    vUv = uv; 
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }}
            `,
            fragmentShader: `
                varying vec2 vUv;
                uniform float u_headY;
                uniform float u_tailY;
                uniform float u_aspect;

                // Standard distance to a line segment
                float sdCapsule( vec2 p, vec2 a, vec2 b, float r ) {{
                    vec2 pa = p - a, ba = b - a;
                    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
                    return length( pa - ba*h ) - r;
                }}

                // Global map function to calculate shape
                float map(vec2 p) {{
                    vec2 head = vec2(0.0, u_headY);
                    vec2 tail = vec2(0.0, u_tailY);
                    
                    // Radius 0.44 fills the width beautifully without clipping
                    return sdCapsule(p, head, tail, 0.44);
                }}

                void main() {{
                    vec2 uv = vUv - 0.5;
                    uv.y *= u_aspect;

                    // Evaluate our distance field
                    float d = map(uv);
                    float alpha = smoothstep(0.015, 0.0, d);

                    // --- FAKE 3D LIGHTING ENGINE ---
                    
                    // 1. Calculate the surface Normal (the 3D slope of the shape)
                    vec2 eps = vec2(0.01, 0.0);
                    vec3 normal = normalize(vec3(
                        map(uv + eps.xy) - map(uv - eps.xy),
                        map(uv + eps.yx) - map(uv - eps.yx),
                        0.08 // Z-thickness of the jelly
                    ));

                    // 2. Light setup (hitting from top-left)
                    vec3 lightDir = normalize(vec3(-1.0, 1.0, 1.5));
                    float diffuse = max(dot(normal, lightDir), 0.0);

                    // 3. Specular highlight (the shiny glass reflection)
                    vec3 viewDir = vec3(0.0, 0.0, 1.0);
                    vec3 halfDir = normalize(lightDir + viewDir);
                    float specular = pow(max(dot(normal, halfDir), 0.0), 32.0);

                    // 4. Colors
                    vec3 shadowColor = vec3(0.75, 0.80, 0.90); // Darker blue/grey edge
                    vec3 highlightColor = vec3(1.0, 1.0, 1.0); // Pure white top
                    
                    vec3 finalJellyColor = mix(shadowColor, highlightColor, diffuse);
                    finalJellyColor += specular * 0.8; // Apply the gloss

                    // 5. Drop shadow (offset slightly down and right)
                    float dShadow = map(uv - vec2(0.02, -0.02));
                    float shadowAlpha = smoothstep(0.15, 0.0, dShadow) * 0.35; // Soft blur

                    // 6. Blend the drop shadow and the 3D jelly shape
                    vec4 shadowLayer = vec4(0.0, 0.0, 0.0, shadowAlpha);
                    vec4 jellyLayer = vec4(finalJellyColor, 0.95); // 95% opaque

                    gl_FragColor = mix(shadowLayer, jellyLayer, alpha);
                }}
            `,
            transparent: true
        }});

        const jMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), jMaterial);
        jScene.add(jMesh);

        function animateJelly() {{
            // 1. Head shoots quickly toward the target
            const headForce = (targetY - headY) * 0.12;
            headVel = (headVel + headForce) * 0.65;
            headY += headVel;

            // 2. Tail drags heavily behind the head (This is the GOO!)
            const tailForce = (headY - tailY) * 0.10;
            tailVel = (tailVel + tailForce) * 0.55;
            tailY += tailVel;

            jMaterial.uniforms.u_headY.value = headY;
            jMaterial.uniforms.u_tailY.value = tailY;

            jRenderer.render(jScene, jCamera);
            requestAnimationFrame(animateJelly);
        }}
        
        setTimeout(() => {{
            const activeBtn = document.querySelector('.icon-btn.active');
            if (activeBtn) {{
                const exactY = getUvY(activeBtn);
                targetY = exactY;
                headY = exactY;
                tailY = exactY; // Snap both parts so it starts round
            }}
            animateJelly();
        }}, 50);
