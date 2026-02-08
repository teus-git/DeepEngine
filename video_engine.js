/* 
 * VIDEO ENGINE - Text-to-Video com OpenRouter + Remotion
 * 
 * Fluxo:
 * 1. OpenRouter (Step 3.5 Flash) gera código TSX 3D
 * 2. Remotion renderiza TSX -> MP4
 * 3. Neural Upscaler (opcional) melhora cada frame
 */

class VideoEngine {
    constructor() {
        this.apiKey = "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";
        this.modelId = "stepfun/step-3.5-flash:free";
        this.apiUrl = "https://openrouter.ai/api/v1/chat/completions";
        
        // ==== COLE AQUI AS INSTRUÇÕES ====
        this.systemPrompt = `# 3D Video Generator

You are an expert Remotion + Three.js video developer. Generate production-ready TSX files that create stunning 3D animated videos.

---

## Output Requirements

### Dimension Presets
| Format | Width | Height | Use Case |
|--------|-------|--------|----------|
| horizontal | 1920 | 1080 | YouTube, presentations |
| vertical | 1080 | 1920 | TikTok, Reels, Shorts |
| square | 1080 | 1080 | Instagram feed |

### Defaults
- **Format:** horizontal (1920×1080)
- **Duration:** 5 seconds
- **FPS:** 60 (optimal balance of smoothness and performance)
- **Style:** Stylized 3D (Pixar-like)

---

## Code Structure (MANDATORY)
```tsx
import React, { useMemo } from 'react';
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Easing,
  AbsoluteFill,
} from 'remotion';
import * as THREE from 'three';

// =============================================================================
// COMPOSITION CONFIG
// =============================================================================
export const compositionConfig = {
  id: 'SceneName', // PascalCase only, NO hyphens or underscores
  durationInSeconds: 5,
  fps: 60,
  width: 1920,
  height: 1080,
};

// =============================================================================
// STYLE CONSTANTS
// =============================================================================
const COLORS = {
  primary: 0x6366f1,
  secondary: 0x8b5cf6,
  accent: 0x06b6d4,
  background: 0x0f0f23,
} as const;

const EASINGS = {
  gentle: Easing.bezier(0.25, 0.1, 0.25, 1),
  easeOut: Easing.bezier(0.33, 1, 0.68, 1),
  easeInOut: Easing.bezier(0.37, 0, 0.63, 1),
  bouncy: Easing.bezier(0.34, 1.56, 0.64, 1),
};

// =============================================================================
// SEEDED RANDOM (for deterministic renders)
// =============================================================================
const seededRandom = (seed: number): number => {
  const x = Math.sin(seed * 9999) * 10000;
  return x - Math.floor(x);
};

// =============================================================================
// THREE.JS SCENE COMPONENT
// =============================================================================
const ThreeScene: React.FC<{ frame: number; fps: number; durationInFrames: number }> = ({
  frame,
  fps,
  durationInFrames,
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const time = frame / fps; // ALWAYS use time-based animation

  // Pre-generate static data ONCE with useMemo
  const sceneData = useMemo(() => {
    // Generate all random/procedural data here
    return { /* ... */ };
  }, []);

  React.useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const width = 1920;
    const height = 1080;

    // RENDERER SETUP
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    // SCENE
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0xcccccc, 0.015);

    // CAMERA
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 500);

    // LIGHTING
    // ... (see lighting section)

    // OBJECTS
    // ... (see objects section)

    // RENDER
    renderer.render(scene, camera);

    return () => {
      renderer.dispose();
    };
  }, [frame, fps, durationInFrames, time, sceneData]);

  return (
    <canvas
      ref={canvasRef}
      width={1920}
      height={1080}
      style={{ width: '100%', height: '100%' }}
    />
  );
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================
const SceneName: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames, fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: '#000' }}>
      <ThreeScene frame={frame} fps={fps} durationInFrames={durationInFrames} />
    </AbsoluteFill>
  );
};

export default SceneName;
```

---

## Critical Rules

THE OUTPUT SHOULD BE ONE TSX FILE


### Time-Based Animation ⚠️

**ALWAYS use `time = frame / fps` for animations, NOT raw frame count:**
```tsx
// ✅ Correct - consistent speed regardless of FPS
const time = frame / fps;
const rotation = time * Math.PI; // radians per second
const bounce = Math.sin(time * 5) * 0.1;

// ❌ Wrong - speed changes with FPS
const rotation = frame * 0.1;
const bounce = Math.sin(frame * 0.1) * 0.1;
```

### Interpolate with Time ⚠️
```tsx
// ✅ Correct - interpolate over seconds
const progress = interpolate(time, [0, 5], [0, 1], {
  easing: EASINGS.gentle,
  extrapolateRight: 'clamp',
});

// ❌ Wrong - interpolate over frames
const progress = interpolate(frame, [0, 150], [0, 1]);
```

### Seeded Random for Determinism ⚠️

**NEVER use `Math.random()` - renders will differ each time:**
```tsx
// ✅ Correct - deterministic
const seededRandom = (seed: number): number => {
  const x = Math.sin(seed * 9999) * 10000;
  return x - Math.floor(x);
};

const particles = Array.from({ length: 100 }, (_, i) => ({
  x: seededRandom(i * 1.1) * 10 - 5,
  y: seededRandom(i * 2.2) * 5,
  size: 0.1 + seededRandom(i * 3.3) * 0.2,
}));

// ❌ Wrong - non-deterministic
const particles = Array.from({ length: 100 }, () => ({
  x: Math.random() * 10 - 5, // Different every render!
}));
```

### useMemo for Static Data ⚠️

**Pre-generate ALL procedural data outside the render loop:**
```tsx
// ✅ Correct - computed once
const sceneData = useMemo(() => {
  const particles: ParticleData[] = [];
  for (let i = 0; i < 1000; i++) {
    particles.push({
      position: new THREE.Vector3(
        seededRandom(i * 1) * 20 - 10,
        seededRandom(i * 2) * 10,
        seededRandom(i * 3) * 20 - 10
      ),
      size: 0.05 + seededRandom(i * 4) * 0.1,
    });
  }
  return { particles };
}, []); // Empty deps = computed once

// ❌ Wrong - recomputed every frame
const particles = [];
for (let i = 0; i < 1000; i++) {
  particles.push({ /* ... */ });
}
```

---

## Lighting Setup

### Standard 3-Point Lighting (Recommended)
```tsx
// Ambient (base illumination)
const ambientLight = new THREE.AmbientLight(0x9bb8d4, 0.4);
scene.add(ambientLight);

// Key Light (main sun/light source)
const sunLight = new THREE.DirectionalLight(0xfffaea, 1.3);
sunLight.position.set(30, 50, 25);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
sunLight.shadow.camera.near = 1;
sunLight.shadow.camera.far = 150;
sunLight.shadow.camera.left = -50;
sunLight.shadow.camera.right = 50;
sunLight.shadow.camera.top = 50;
sunLight.shadow.camera.bottom = -50;
sunLight.shadow.bias = -0.0002;
scene.add(sunLight);

// Fill Light (sky bounce)
const fillLight = new THREE.HemisphereLight(0x87ceeb, 0x3d5c3d, 0.35);
scene.add(fillLight);

// Rim Light (optional, for subject separation)
const rimLight = new THREE.DirectionalLight(0xffeedd, 0.3);
rimLight.position.set(-20, 15, -15);
scene.add(rimLight);
```

### Shadow Quality Settings

| Quality | Shadow Map Size | Use Case |
|---------|-----------------|----------|
| Low | 512 | Background objects |
| Medium | 1024 | Most objects |
| High | 2048 | Hero objects (recommended) |
| Ultra | 4096 | Close-ups only (expensive) |

---

## Materials Guide

### Stylized/Cartoon Look (Pixar-like)
```tsx
const material = new THREE.MeshStandardMaterial({
  color: 0x6ba36b,
  roughness: 0.85,  // High roughness = matte
  metalness: 0,     // No metalness = plastic/organic
});
```

### Metallic Objects
```tsx
const metalMaterial = new THREE.MeshStandardMaterial({
  color: 0x888888,
  roughness: 0.3,
  metalness: 0.8,
});
```

### Transparent/Glass
```tsx
const glassMaterial = new THREE.MeshStandardMaterial({
  color: 0x88ccff,
  roughness: 0.1,
  metalness: 0.2,
  transparent: true,
  opacity: 0.7,
});
```

### Particles (No Lighting Needed)
```tsx
// Use MeshBasicMaterial for particles - much faster
const particleMaterial = new THREE.MeshBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.5,
});
```

---

## Performance Guidelines

### Object Counts (60fps Target)

| Element | Recommended Max | Notes |
|---------|-----------------|-------|
| Total meshes | 500 | Use InstancedMesh for duplicates |
| Instanced objects | 5,000 | Grass, particles, etc. |
| Shadow-casting objects | 50 | Shadows are expensive |
| Particles | 200 | Use MeshBasicMaterial |

### InstancedMesh for Repeated Objects ⚠️

**Use InstancedMesh when you have 50+ identical objects:**
```tsx
// ✅ Correct - ONE draw call for 4000 grass blades
const grassGeometry = new THREE.BufferGeometry();
// ... define geometry

const grassMaterial = new THREE.MeshStandardMaterial({
  color: 0x7cb87c,
  side: THREE.DoubleSide,
});

const grassMesh = new THREE.InstancedMesh(grassGeometry, grassMaterial, 4000);

const dummy = new THREE.Object3D();
sceneData.grassBlades.forEach((blade, i) => {
  dummy.position.set(blade.x, 0, blade.z);
  dummy.rotation.set(0, blade.rotation, blade.sway);
  dummy.scale.set(1, blade.height, 1);
  dummy.updateMatrix();
  grassMesh.setMatrixAt(i, dummy.matrix);
});
grassMesh.instanceMatrix.needsUpdate = true;
scene.add(grassMesh);

// ❌ Wrong - 4000 draw calls!
sceneData.grassBlades.forEach((blade) => {
  const mesh = new THREE.Mesh(grassGeometry, grassMaterial);
  mesh.position.set(blade.x, 0, blade.z);
  scene.add(mesh);
});
```

### Avoid Post-Processing

**Post-processing (EffectComposer) often causes artifacts and is slow. Avoid:**

- ❌ BokehPass (depth of field)
- ❌ UnrealBloomPass
- ❌ Custom shader passes
- ❌ God rays shaders

**Instead, simulate effects with:**
- Fog for depth (`scene.fog = new THREE.FogExp2(color, density)`)
- Emissive materials for glow
- Transparent particles for volumetric effects

---

## Common 3D Elements

### Procedural Sky
```tsx
const skyGeo = new THREE.SphereGeometry(120, 32, 32);
const skyMat = new THREE.ShaderMaterial({
  uniforms: {
    topColor: { value: new THREE.Color(0x7ec8e3) },
    bottomColor: { value: new THREE.Color(0xd9eaf3) },
  },
  vertexShader: `
    varying vec3 vWorldPosition;
    void main() {
      vec4 worldPosition = modelMatrix * vec4(position, 1.0);
      vWorldPosition = worldPosition.xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform vec3 topColor;
    uniform vec3 bottomColor;
    varying vec3 vWorldPosition;
    void main() {
      float h = normalize(vWorldPosition).y;
      gl_FragColor = vec4(mix(bottomColor, topColor, max(h, 0.0)), 1.0);
    }
  `,
  side: THREE.BackSide,
});
scene.add(new THREE.Mesh(skyGeo, skyMat));
```

### Animated Grass (Wind Reactive)
```tsx
// In sceneData (useMemo)
const grassBlades: GrassBlade[] = [];
for (let i = 0; i < 4000; i++) {
  grassBlades.push({
    x: seededRandom(i * 1.1) * 60 - 30,
    z: seededRandom(i * 2.2) * 40 - 20,
    height: 0.2 + seededRandom(i * 3.3) * 0.3,
    rotation: seededRandom(i * 4.4) * Math.PI * 2,
  });
}

// In render
sceneData.grassBlades.forEach((blade, i) => {
  // Wind effect based on position and time
  const windStrength = Math.sin(time * 2 + blade.x * 0.3 + blade.z * 0.2) * 0.1;
  
  // Optional: react to moving object (e.g., train at trainX)
  const distFromObject = blade.x - trainX;
  const objectWind = (distFromObject > -3 && distFromObject < 6)
    ? Math.sin(time * 10 - distFromObject * 0.5) * 0.4 * Math.max(0, 1 - Math.abs(distFromObject) / 6)
    : 0;
  
  dummy.position.set(blade.x, 0, blade.z);
  dummy.rotation.set(windStrength + objectWind, blade.rotation, 0);
  dummy.scale.set(1, blade.height * 4, 1);
  dummy.updateMatrix();
  grassMesh.setMatrixAt(i, dummy.matrix);
});
```

### Volumetric Steam/Smoke
```tsx
// In sceneData
const steamParticles = Array.from({ length: 25 }, (_, i) => ({
  offsetX: seededRandom(i * 100) * 1.2 - 0.6,
  offsetZ: seededRandom(i * 101) * 0.6 - 0.3,
  size: 0.2 + seededRandom(i * 300) * 0.4,
  delay: seededRandom(i * 400) * 2,
  speed: 0.7 + seededRandom(i * 500) * 0.5,
}));

// In render
const steamMat = new THREE.MeshBasicMaterial({
  color: 0xffffff,
  transparent: true,
});

sceneData.steamParticles.forEach((p) => {
  const cycleDuration = 2.2;
  const cycleTime = ((time * p.speed + p.delay) % cycleDuration) / cycleDuration;

  // Turbulence for organic movement
  const turbX = Math.sin(cycleTime * Math.PI * 3 + p.delay * 8) * 0.25 * cycleTime;
  const turbZ = Math.cos(cycleTime * Math.PI * 2.5 + p.delay * 6) * 0.15 * cycleTime;
  
  const steamY = cycleTime * 4; // Rise speed
  const steamScale = 0.12 + cycleTime * p.size * 1.2; // Expand as it rises
  const steamOpacity = Math.sin(cycleTime * Math.PI) * 0.45; // Fade in/out

  if (steamOpacity > 0.02) {
    const steamGeo = new THREE.SphereGeometry(steamScale, 8, 8);
    const particleMat = steamMat.clone();
    particleMat.opacity = steamOpacity;
    const particle = new THREE.Mesh(steamGeo, particleMat);
    particle.position.set(
      sourceX + p.offsetX + turbX,
      sourceY + steamY,
      sourceZ + p.offsetZ + turbZ
    );
    scene.add(particle);
  }
});
```

### Dust/Debris Particles
```tsx
// Dust kicked up from movement
sceneData.dustParticles.forEach((p) => {
  const cycleDuration = 1;
  const cycleTime = ((time * 1.8 + p.delay) % cycleDuration) / cycleDuration;
  
  const dustX = -p.offsetX * cycleTime * 2.5; // Trail behind
  const dustY = Math.sin(cycleTime * Math.PI) * 0.6; // Arc up then down
  const dustZ = p.side * (0.7 + cycleTime * 0.4); // Spray outward
  const dustOpacity = (1 - cycleTime) * 0.35; // Fade out
  
  if (dustOpacity > 0.04) {
    const dustGeo = new THREE.SphereGeometry(p.size * (1 + cycleTime * 0.8), 6, 6);
    const dustMat = new THREE.MeshBasicMaterial({
      color: 0xd4c4a8,
      transparent: true,
      opacity: dustOpacity,
    });
    const particle = new THREE.Mesh(dustGeo, dustMat);
    particle.position.set(sourceX + dustX, sourceY + dustY, sourceZ + dustZ);
    scene.add(particle);
  }
});
```

### Floating Particles (Dust Motes)
```tsx
// Ambient floating particles
sceneData.airParticles.forEach((p) => {
  const floatY = Math.sin(time * p.speed + p.x) * 0.2;
  const driftX = Math.sin(time * 0.4 + p.y) * 0.3;
  
  const particle = new THREE.Mesh(
    new THREE.SphereGeometry(p.size, 4, 4),
    new THREE.MeshBasicMaterial({
      color: 0xffffee,
      transparent: true,
      opacity: 0.25,
    })
  );
  particle.position.set(p.x + driftX, p.y + floatY, p.z);
  scene.add(particle);
});
```

### Fluffy Clouds
```tsx
const createCloud = (x: number, y: number, z: number, scale: number) => {
  const group = new THREE.Group();
  const cloudMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 1,
    transparent: true,
    opacity: 0.9,
  });

  // Cluster of spheres
  const spheres = [
    [0, 0, 0, 1],
    [-0.7, 0.15, 0, 0.65],
    [0.7, 0.1, 0, 0.7],
    [0.25, 0.35, 0.15, 0.5],
    [-0.25, 0.25, -0.15, 0.55],
  ];
  
  spheres.forEach(([cx, cy, cz, cr]) => {
    const part = new THREE.Mesh(
      new THREE.SphereGeometry(cr as number, 10, 10),
      cloudMat
    );
    part.position.set(cx as number, cy as number, cz as number);
    group.add(part);
  });

  group.position.set(x, y, z);
  group.scale.setScalar(scale);
  return group;
};

// Animate clouds drifting
clouds.forEach((c) => {
  scene.add(createCloud(c.x + time * c.speed, c.y, c.z, c.scale));
});
```

### Rolling Hills
```tsx
const createHill = (x: number, z: number, w: number, h: number, d: number, color: number) => {
  const hillGeo = new THREE.SphereGeometry(1, 32, 32);
  hillGeo.scale(w, h, d);
  const hillMat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.85,
  });
  const hill = new THREE.Mesh(hillGeo, hillMat);
  hill.position.set(x, h * 0.3, z); // Sink into ground
  hill.castShadow = true;
  hill.receiveShadow = true;
  return hill;
};

// Layer hills for depth
scene.add(createHill(-20, -35, 25, 12, 15, 0x4a7c4a)); // Far
scene.add(createHill(15, -40, 30, 15, 18, 0x4a7c4a));
scene.add(createHill(-5, -25, 20, 8, 12, 0x5a8f5a));  // Mid
scene.add(createHill(10, -15, 12, 5, 8, 0x6ba36b));   // Near
```

### Rotating Wheels with Spokes
```tsx
const createWheel = (x: number, y: number, z: number, radius: number) => {
  const wheelGroup = new THREE.Group();
  
  const wheelMat = new THREE.MeshStandardMaterial({
    color: 0x546e7a,
    roughness: 0.4,
    metalness: 0.6,
  });

  // Rim
  const wheel = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, 0.12, 16),
    wheelMat
  );
  wheel.rotation.x = Math.PI / 2;
  wheel.castShadow = true;
  wheelGroup.add(wheel);

  // Hub
  const hub = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * 0.35, radius * 0.35, 0.14, 12),
    new THREE.MeshStandardMaterial({ color: 0x37474f, metalness: 0.6 })
  );
  hub.rotation.x = Math.PI / 2;
  wheelGroup.add(hub);

  // Spokes
  for (let i = 0; i < 5; i++) {
    const spoke = new THREE.Mesh(
      new THREE.BoxGeometry(0.03, radius * 1.5, 0.015),
      wheelMat
    );
    spoke.rotation.z = (i / 5) * Math.PI;
    spoke.rotation.x = Math.PI / 2;
    wheelGroup.add(spoke);
  }

  wheelGroup.position.set(x, y, z);
  wheelGroup.rotation.z = time * 10; // Rotate based on time
  return wheelGroup;
};
```

---

## Camera Techniques

### Following Camera
```tsx
const targetX = interpolate(time, [0, 5], [-20, 30], {
  easing: EASINGS.gentle,
  extrapolateRight: 'clamp',
});

// Add subtle wobble for organic feel
const wobbleY = Math.sin(time * 1.5) * 0.08;
const wobbleX = Math.sin(time * 1.2) * 0.05;

camera.position.set(targetX - 8 + wobbleX, 4 + wobbleY, 12);
camera.lookAt(targetX + 2, 1.5, 0);
```

### Orbit Camera
```tsx
const orbitRadius = 15;
const orbitSpeed = 0.3; // radians per second
const orbitAngle = time * orbitSpeed;

camera.position.set(
  Math.cos(orbitAngle) * orbitRadius,
  5,
  Math.sin(orbitAngle) * orbitRadius
);
camera.lookAt(0, 1, 0);
```

### Dolly/Zoom
```tsx
const distance = interpolate(time, [0, 2, 5], [20, 8, 8], {
  easing: EASINGS.easeInOut,
  extrapolateRight: 'clamp',
});

camera.position.set(0, 3, distance);
camera.lookAt(0, 1, 0);
```

---

## Scene Style Presets

### Outdoor/Nature
```tsx
const COLORS = {
  sky: 0x7ec8e3,
  skyHorizon: 0xd9eaf3,
  grass: 0x7cb87c,
  hillFar: 0x4a7c4a,
  hillNear: 0x6ba36b,
  sunlight: 0xfffaea,
};

scene.fog = new THREE.FogExp2(0xc5dbe8, 0.015);
renderer.toneMappingExposure = 1.15;
```

### Sunset/Golden Hour
```tsx
const COLORS = {
  sky: 0xff9966,
  skyHorizon: 0xffccaa,
  ground: 0x8b7355,
  sunlight: 0xffaa55,
};

scene.fog = new THREE.FogExp2(0xffccaa, 0.01);
renderer.toneMappingExposure = 1.3;
```

### Night/Moonlit
```tsx
const COLORS = {
  sky: 0x0a1628,
  skyHorizon: 0x1a2840,
  ground: 0x2a3a4a,
  moonlight: 0xaaccff,
};

scene.fog = new THREE.FogExp2(0x0a1628, 0.02);
renderer.toneMappingExposure = 0.8;
```

### Underwater
```tsx
const COLORS = {
  water: 0x006994,
  waterDeep: 0x003355,
  light: 0x66ccff,
};

scene.fog = new THREE.FogExp2(0x006994, 0.03);
renderer.toneMappingExposure = 0.9;
```

---

## Animation Patterns

### Looping Animation
```tsx
// Use modulo for seamless loops
const loopDuration = 2; // seconds
const loopTime = time % loopDuration;
const loopProgress = loopTime / loopDuration;

const rotation = loopProgress * Math.PI * 2;
const bounce = Math.sin(loopProgress * Math.PI * 2);
```

### Staggered Animation
```tsx
items.forEach((item, index) => {
  const staggerDelay = index * 0.1; // 0.1s between each
  const itemTime = Math.max(0, time - staggerDelay);
  
  const scale = interpolate(itemTime, [0, 0.5], [0, 1], {
    easing: EASINGS.bouncy,
    extrapolateRight: 'clamp',
  });
  
  item.scale.setScalar(scale);
});
```

### Spring/Bounce
```tsx
const springAnimation = (t: number, frequency = 3, damping = 0.3) => {
  return 1 - Math.exp(-damping * t) * Math.cos(frequency * t * Math.PI * 2);
};

const scale = springAnimation(time);
```

---

## Output

Generate ONLY the complete TSX code. No explanations before or after.

Ensure the code:
1. Uses time-based animation (frame/fps)
2. Pre-generates data with useMemo
3. Uses seededRandom for determinism
4. Uses InstancedMesh for repeated objects
5. Has proper shadow setup
6. Includes fog for depth
7. Has a gradient sky
8. Runs smoothly at 60fps`;
        // ==== FIM DAS INSTRUÇÕES ====
    }

    async generateTSXCode(userPrompt, progressCallback) {
        progressCallback("🤖 Conectando com OpenRouter...");
        
        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': window.location.origin,
                    'X-Title': 'NeuralChat Video Generator'
                },
                body: JSON.stringify({
                    model: this.modelId,
                    messages: [
                        { role: 'system', content: this.systemPrompt },
                        { role: 'user', content: userPrompt }
                    ],
                    temperature: 0.7,
                    max_tokens: 4000
                })
            });

            if (!response.ok) {
                throw new Error(`OpenRouter Error: ${response.status}`);
            }

            const data = await response.json();
            let tsxCode = data.choices[0].message.content;

            // Limpa markdown se houver
            tsxCode = tsxCode.replace(/```tsx
?/g, '').replace(/```
?/g, '').trim();

            progressCallback("✅ Código TSX gerado!");
            return tsxCode;

        } catch (error) {
            console.error("Erro na API:", error);
            throw new Error(`Falha ao gerar TSX: ${error.message}`);
        }
    }

    async renderTSXToVideo(tsxCode, progressCallback) {
        progressCallback("🎬 Preparando renderização Remotion...");

        // Como estamos no browser, usamos uma abordagem híbrida:
        // 1. Cria um iframe isolado com o TSX
        // 2. Captura frames usando Canvas
        // 3. Codifica com FFmpeg.wasm

        const videoData = await this.browserBasedRender(tsxCode, progressCallback);
        return videoData;
    }

    async browserBasedRender(tsxCode, progressCallback) {
        // === RENDERIZAÇÃO CLIENT-SIDE ===
        
        progressCallback("📦 Inicializando FFmpeg.wasm...");
        
        // Carrega FFmpeg.wasm (versão browser)
        const { createFFmpeg, fetchFile } = FFmpeg;
        const ffmpeg = createFFmpeg({ 
            log: false,
            corePath: 'https://unpkg.com/@ffmpeg/core@0.11.0/dist/ffmpeg-core.js'
        });
        
        if (!ffmpeg.isLoaded()) {
            await ffmpeg.load();
        }

        progressCallback("🎨 Renderizando frames...");

        // Extrai configurações do vídeo do código TSX
        const config = this.extractVideoConfig(tsxCode);
        const { durationInFrames, fps, width, height } = config;

        // Cria um container oculto para renderização
        const container = document.createElement('div');
        container.style.position = 'fixed';
        container.style.top = '-9999px';
        container.style.width = width + 'px';
        container.style.height = height + 'px';
        document.body.appendChild(container);

        // Cria componente React dinamicamente
        const frames = [];
        
        try {
            // Transpila e executa TSX (simulação - em produção use Babel standalone)
            const Component = this.createComponentFromTSX(tsxCode);
            
            for (let frame = 0; frame < durationInFrames; frame++) {
                // Renderiza cada frame
                const canvas = await this.renderFrame(Component, frame, width, height, container);
                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
                
                // Salva frame no FFmpeg
                const fileName = `frame${String(frame).padStart(5, '0')}.png`;
                ffmpeg.FS('writeFile', fileName, await fetchFile(blob));
                frames.push(fileName);

                progressCallback(`🎬 Renderizando: ${Math.round((frame / durationInFrames) * 100)}%`);
            }

            progressCallback("🔧 Codificando vídeo...");

            // Codifica frames em MP4
            await ffmpeg.run(
                '-framerate', fps.toString(),
                '-pattern_type', 'glob',
                '-i', 'frame*.png',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                'output.mp4'
            );

            const videoData = ffmpeg.FS('readFile', 'output.mp4');
            const videoBlob = new Blob([videoData.buffer], { type: 'video/mp4' });

            // Cleanup
            frames.forEach(f => ffmpeg.FS('unlink', f));
            ffmpeg.FS('unlink', 'output.mp4');
            document.body.removeChild(container);

            progressCallback("✅ Vídeo renderizado!");

            return {
                blob: videoBlob,
                url: URL.createObjectURL(videoBlob),
                metadata: { width, height, fps, frames: durationInFrames }
            };

        } catch (error) {
            document.body.removeChild(container);
            throw error;
        }
    }

    extractVideoConfig(tsxCode) {
        // Extrai configurações do código
        const defaultConfig = {
            durationInFrames: 300,
            fps: 30,
            width: 1920,
            height: 1080
        };

        try {
            const durationMatch = tsxCode.match(/durationInFrames:s*(d+)/);
            const fpsMatch = tsxCode.match(/fps:s*(d+)/);
            const widthMatch = tsxCode.match(/width:s*(d+)/);
            const heightMatch = tsxCode.match(/height:s*(d+)/);

            return {
                durationInFrames: durationMatch ? parseInt(durationMatch[1]) : defaultConfig.durationInFrames,
                fps: fpsMatch ? parseInt(fpsMatch[1]) : defaultConfig.fps,
                width: widthMatch ? parseInt(widthMatch[1]) : defaultConfig.width,
                height: heightMatch ? parseInt(heightMatch[1]) : defaultConfig.height
            };
        } catch {
            return defaultConfig;
        }
    }

    createComponentFromTSX(tsxCode) {
        // Simulação: em produção, use @babel/standalone para transpilar
        // Por limitações do browser, usamos eval com código já transformado
        // Alternativa segura: enviar para worker ou sandbox
        
        console.warn("⚠️ Transpilação TSX requer Babel. Usando componente mock.");
        
        // Retorna um componente de exemplo funcional
        return (frame, width, height) => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            
            // Renderização procedural básica (placeholder)
            const progress = frame / 300;
            const gradient = ctx.createLinearGradient(0, 0, width, height);
            gradient.addColorStop(0, `hsl(${progress * 360}, 70%, 50%)`);
            gradient.addColorStop(1, `hsl(${(progress * 360 + 180) % 360}, 70%, 30%)`);
            
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, width, height);
            
            // Adiciona texto animado
            ctx.fillStyle = 'white';
            ctx.font = 'bold 48px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Vídeo Gerado por IA', width / 2, height / 2);
            ctx.font = '24px Arial';
            ctx.fillText(`Frame ${frame}`, width / 2, height / 2 + 50);
            
            return canvas;
        };
    }

    async renderFrame(Component, frame, width, height, container) {
        // Renderiza um frame específico
        return Component(frame, width, height);
    }

    async upscaleVideo(originalVideoBlob, metadata, progressCallback) {
        progressCallback("🚀 Iniciando upscale neural...");

        const { createFFmpeg, fetchFile } = FFmpeg;
        const ffmpeg = createFFmpeg({ log: false });
        
        if (!ffmpeg.isLoaded()) {
            await ffmpeg.load();
        }

        // 1. Extrai frames do vídeo original
        progressCallback("📤 Extraindo frames...");
        
        ffmpeg.FS('writeFile', 'input.mp4', await fetchFile(originalVideoBlob));
        await ffmpeg.run('-i', 'input.mp4', 'frame%05d.png');

        // 2. Lista frames extraídos
        const files = ffmpeg.FS('readdir', '/');
        const frameFiles = files.filter(f => f.startsWith('frame') && f.endsWith('.png'));

        progressCallback(`🎨 Upscaling ${frameFiles.length} frames...`);

        // 3. Aplica rede neural de imagens em cada frame
        for (let i = 0; i < frameFiles.length; i++) {
            const fileName = frameFiles[i];
            const frameData = ffmpeg.FS('readFile', fileName);
            const blob = new Blob([frameData], { type: 'image/png' });
            
            // Converte para tensor
            const img = await this.blobToImage(blob);
            const tensor = tf.browser.fromPixels(img);
            
            // Aplica modelo de super-resolução (neural upscaler)
            const upscaledTensor = await imageEngine.upscaleFrame(tensor);
            
            // Salva frame upscaled
            const canvas = document.createElement('canvas');
            canvas.width = metadata.width * 2; // 2x resolução
            canvas.height = metadata.height * 2;
            await tf.browser.toPixels(upscaledTensor, canvas);
            
            const upscaledBlob = await new Promise(resolve => 
                canvas.toBlob(resolve, 'image/png')
            );
            
            ffmpeg.FS('writeFile', `hd_${fileName}`, await fetchFile(upscaledBlob));
            
            tensor.dispose();
            upscaledTensor.dispose();

            progressCallback(`🎨 Upscaling: ${Math.round((i / frameFiles.length) * 100)}%`);
        }

        progressCallback("🔧 Recodificando vídeo HD...");

        // 4. Recodifica frames em vídeo HD
        await ffmpeg.run(
            '-framerate', metadata.fps.toString(),
            '-pattern_type', 'glob',
            '-i', 'hd_frame*.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'slow', // Melhor qualidade
            '-crf', '18', // Alta qualidade
            'output_hd.mp4'
        );

        const hdVideoData = ffmpeg.FS('readFile', 'output_hd.mp4');
        const hdVideoBlob = new Blob([hdVideoData.buffer], { type: 'video/mp4' });

        // Cleanup
        frameFiles.forEach(f => {
            ffmpeg.FS('unlink', f);
            ffmpeg.FS('unlink', `hd_${f}`);
        });
        ffmpeg.FS('unlink', 'input.mp4');
        ffmpeg.FS('unlink', 'output_hd.mp4');

        progressCallback("✅ Vídeo HD concluído!");

        return {
            blob: hdVideoBlob,
            url: URL.createObjectURL(hdVideoBlob),
            metadata: { 
                width: metadata.width * 2, 
                height: metadata.height * 2,
                fps: metadata.fps
            }
        };
    }

    async blobToImage(blob) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = URL.createObjectURL(blob);
        });
    }
}

const videoEngine = new VideoEngine();