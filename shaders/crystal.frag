#version 330 core

out vec4 fragColor;

uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uCameraPosition;

uniform float uCrystalRadius;
uniform float uRefractionRatio;

uniform vec3 uLightPos;
uniform vec3 uSkyLow;
uniform vec3 uSkyHigh;
uniform vec3 uSunColor;
uniform float uSkyHorizonScale;
uniform float uSkyHorizonBias;
uniform float uSunGlowStrength;
uniform float uSunGlowPower;
uniform float uSunCoreStrength;
uniform float uSunCorePower;

uniform vec3 uFractalOffset;
uniform float uFractalOffsetScale;
uniform float uFieldSeedDivisor;
uniform float uFieldWeightDecay;
uniform float uFieldResponseExponent;
uniform float uFieldContrast;
uniform float uFieldDensityBias;

uniform vec3 uDriftSpeed;
uniform float uDomain1Scale;
uniform vec3 uDomain1Offset;
uniform float uDomain1CameraInfluence;
uniform float uDomain1DriftInfluence;
uniform float uDomain2Scale;
uniform float uDomain2ScaleOscAmplitude;
uniform float uDomain2ScaleOscSpeed;
uniform vec3 uDomain2Offset;
uniform float uDomain2CameraInfluence;
uniform float uDomain2DriftInfluence;

uniform float uFieldSeed1;
uniform int uFieldIterations1;
uniform float uFieldStrength1;
uniform float uFieldSeed2;
uniform int uFieldIterations2;
uniform float uFieldStrength2;
uniform vec3 uLayer1Weights;
uniform vec3 uLayer1Powers;
uniform vec3 uLayer2Weights;
uniform vec3 uLayer2Powers;

uniform float uHashScale;
uniform float uHashOffset;

uniform int uVolumeSteps;
uniform float uVolumeSampleOffset;
uniform float uInnerMaskStart;
uniform float uInnerMaskEnd;
uniform float uNebulaScale;
uniform vec3 uDensityWeights;
uniform float uDensityScale;
uniform float uStarGridScale;
uniform float uStarThreshold;
uniform float uStarCameraInfluence;
uniform vec3 uStarColor;
uniform float uStarIntensity;
uniform float uNebulaEmissionStrength;
uniform float uAbsorptionDensityScale;
uniform float uAbsorptionBase;
uniform float uVolumeEmissionGain;
uniform float uTransmittanceCutoff;
uniform vec3 uDeepTint;
uniform float uDeepTintStrength;

uniform float uFresnelBase;
uniform float uFresnelScale;
uniform float uFresnelPower;
uniform float uSurfaceBias;
uniform float uExitSkyStrength;
uniform float uSpecularPower;
uniform float uSpecularStrength;
uniform float uRimPower;
uniform float uRimStrength;
uniform vec3 uCrystalTint;

uniform vec3 uCameraBase;
uniform float uCameraBaseScale;
uniform vec3 uCameraOffset;
uniform float uCameraYawBase;
uniform float uCameraYawAmplitude;
uniform float uCameraYawSpeed;
uniform float uCameraPitchBase;
uniform float uCameraPitchAmplitude;
uniform float uCameraPitchSpeed;
uniform vec3 uCameraTarget;
uniform float uFocalLength;

uniform float uToneExposure;
uniform float uGamma;
uniform float uVignetteMin;
uniform float uVignetteScale;
uniform float uVignettePower;

const float PI = 3.14159265358979323846;
const float MISS = 1.0e4;
const float EPSILON = 1.0e-6;
const int FIELD_MAX_ITERATIONS = 64;
const int VOLUME_MAX_STEPS = 128;

mat2 rotation(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, s, -s, c);
}

float saturate(float value)
{
    return clamp(value, 0.0, 1.0);
}

// check if ray hits sphere
// https://iquilezles.org/articles/intersectors/
// ro - ray origin
// rd - ray distance
// h - hit
// b, c - math bs
// t0, t1 - distance to hits
float raySphere(vec3 ro, vec3 rd, float radius)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;

    if (h < 0.0)
    {
        return MISS;
    }

    h = sqrt(h);
    float t0 = -b - h;
    float t1 = -b + h;

    if (t0 > 0.0)
    {
        return t0;
    }

    if (t1 > 0.0)
    {
        return t1;
    }

    return MISS;
}

// same as above but if ray is inside sphere
// return distance if exit is front of ray
float innerRaySphere(vec3 ro, vec3 rd, float radius)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;

    if (h < 0.0)
    {
        return MISS;
    }

    h = sqrt(h);
    float t = -b + h;
    return t > 0.0 ? t : MISS;
}

// get where sphere position is facing towards
vec3 sphereNormal(vec3 position)
{
    return normalize(position);
}

// nebula
// https://www.shadertoy.com/view/cdj3DW
float field(vec3 p, float seed, int iterations, float strength)
{
    float accum = seed / max(uFieldSeedDivisor, EPSILON);
    float previousMagnitude = 0.0;
    float totalWeight = 0.0;

    for (int i = 0; i < FIELD_MAX_ITERATIONS; ++i)
    {
        if (i >= iterations)
        {
            break;
        }

        float magnitude = max(dot(p, p), EPSILON);
        p = abs(p) / magnitude + uFractalOffset * uFractalOffsetScale;

        float decay = max(uFieldWeightDecay, EPSILON);
        float weight = exp(-float(i) / decay);
        float response = exp(-strength * pow(abs(magnitude - previousMagnitude), max(uFieldResponseExponent, EPSILON)));

        accum += weight * response;
        totalWeight += weight;
        previousMagnitude = magnitude;
    }

    return max(0.0, uFieldContrast * accum / max(totalWeight, EPSILON) - uFieldDensityBias);
}

// pseudo random bs
// input vec 3
// output - number 0-1
float hash31(vec3 p)
{
    p = fract(p * uHashScale);
    p += dot(p, p.yzx + uHashOffset);
    return fract((p.x + p.y) * p.z);
}

// get rgb from t
vec3 poweredLayer(float t, vec3 weights, vec3 powers)
{
    float v = max(t, 0.0);
    return weights * vec3(
        pow(v, max(powers.x, EPSILON)),
        pow(v, max(powers.y, EPSILON)),
        pow(v, max(powers.z, EPSILON))
    );
}

// how should nebula look like at this position
vec3 nebulaDensityColor(vec3 p)
{
    vec3 drift = sin(vec3(uTime) * uDriftSpeed);

    vec3 domain1 = p * uDomain1Scale + uDomain1Offset + uCameraPosition * uDomain1CameraInfluence + drift * uDomain1DriftInfluence;

    float animatedDomain2Scale = uDomain2Scale + uDomain2ScaleOscAmplitude * sin(uTime * uDomain2ScaleOscSpeed);

    vec3 domain2 = p * animatedDomain2Scale + uDomain2Offset + uCameraPosition * uDomain2CameraInfluence + drift * uDomain2DriftInfluence;

    float t1 = field(domain1, uFieldSeed1, uFieldIterations1, uFieldStrength1);
    float t2 = field(domain2, uFieldSeed2, uFieldIterations2, uFieldStrength2);

    return poweredLayer(t1, uLayer1Weights, uLayer1Powers) + poweredLayer(t2, uLayer2Weights, uLayer2Powers);
}

vec3 skyColor(vec3 rd)
{
    float horizon = saturate(rd.y * uSkyHorizonScale + uSkyHorizonBias);
    vec3 sky = mix(uSkyLow, uSkyHigh, horizon);

    vec3 sunDirection = normalize(uLightPos);
    float sun = max(dot(rd, sunDirection), 0.0);
    sky += uSunColor * (uSunGlowStrength * pow(sun, max(uSunGlowPower, EPSILON)) + uSunCoreStrength * pow(sun, max(uSunCorePower, EPSILON)));

    return sky;
}

// glowy bullshit inside the crystal
// at each step - check nebula, check stars, add light, block light
// return that
// vars:
// stepCount - how many samples inside crystal
// stepLength - distance between samples
// accumulated - color collected so far
// transmittance - how much light can still get through
// p - current point inside crystal
// densityColor - nebula color here
// star - is there a star here?
// emission - light produced here
// absorption - how much light gets blocked
vec3 marchCrystalVolume(vec3 ro, vec3 rd, float distanceInside)
{
    int stepCount = clamp(uVolumeSteps, 1, VOLUME_MAX_STEPS);
    float stepLength = distanceInside / float(stepCount);
    vec3 accumulated = vec3(0.0);
    float transmittance = 1.0;

    float maskStart = min(uInnerMaskStart, uInnerMaskEnd);
    float maskEnd = max(max(uInnerMaskStart, uInnerMaskEnd), maskStart + EPSILON);

    for (int i = 0; i < VOLUME_MAX_STEPS; ++i)
    {
        if (i >= stepCount)
        {
            break;
        }

        float t = (float(i) + uVolumeSampleOffset) * stepLength;
        vec3 p = ro + rd * t;
        float radialDistance = length(p);

        float innerMask = 1.0 - smoothstep(uCrystalRadius * maskStart, uCrystalRadius * maskEnd, radialDistance);
        vec3 densityColor = nebulaDensityColor(p * uNebulaScale) * innerMask;
        float density = saturate(dot(densityColor, uDensityWeights) * uDensityScale);

        vec3 cell = floor((p + uCameraPosition * uStarCameraInfluence) * uStarGridScale);
        float star = step(uStarThreshold, hash31(cell)) * innerMask;

        vec3 emission = densityColor * uNebulaEmissionStrength + star * uStarColor * uStarIntensity;

        float absorption = density * uAbsorptionDensityScale + uAbsorptionBase;
        float alpha = 1.0 - exp(-max(absorption, 0.0) * stepLength);

        accumulated += transmittance * emission * alpha * uVolumeEmissionGain;
        transmittance *= 1.0 - alpha;

        if (transmittance < uTransmittanceCutoff)
        {
            break;
        }
    }

    accumulated += uDeepTint * (1.0 - transmittance) * uDeepTintStrength;
    return accumulated;
}

// render the damn crystal using all that bs above
// shoot ray - if miss - draw sky
// if hit - reflect light, do it again but through the crystal, add nebula, add shiny bs, add glow edge
// return final color
// vars
// hitT - where ray hits crystal
// hitPosition - exact hit point
// normal - which way surface faces
// reflected - bounced ray
// refracted - ray going through crystal
// fresnel - how reflective the edge is
// volumeColor - nebula/stars inside
// specular - shiny light spot
// rim - glowing edge
// color - final pixel color
vec3 render(vec3 ro, vec3 rd)
{
    float hitT = raySphere(ro, rd, uCrystalRadius);

    if (hitT >= MISS)
    {
        return skyColor(rd);
    }

    vec3 hitPosition = ro + rd * hitT;
    vec3 normal = sphereNormal(hitPosition);
    vec3 reflected = reflect(rd, normal);
    vec3 refracted = refract(rd, normal, uRefractionRatio);

    float cosTheta = saturate(dot(-rd, normal));
    float fresnel = uFresnelBase + uFresnelScale * pow(1.0 - cosTheta, max(uFresnelPower, EPSILON));
    fresnel = saturate(fresnel);

    vec3 reflectionColor = skyColor(reflected);
    vec3 volumeColor = vec3(0.0);

    if (dot(refracted, refracted) > EPSILON)
    {
        vec3 insideOrigin = hitPosition + refracted * uSurfaceBias;
        float insideDistance = innerRaySphere(insideOrigin, refracted, uCrystalRadius);

        if (insideDistance < MISS)
        {
            volumeColor = marchCrystalVolume(insideOrigin, refracted, insideDistance);

            vec3 exitPosition = insideOrigin + refracted * insideDistance;
            vec3 exitNormal = -sphereNormal(exitPosition);
            vec3 exitRay = refract(refracted, exitNormal, 1.0 / max(uRefractionRatio, EPSILON));

            if (dot(exitRay, exitRay) > EPSILON)
            {
                volumeColor += skyColor(exitRay) * uExitSkyStrength;
            }
        }
    }

    vec3 lightDirection = normalize(uLightPos - hitPosition);
    vec3 halfVector = normalize(lightDirection - rd);
    float specular = pow(max(dot(normal, halfVector), 0.0), max(uSpecularPower, EPSILON));
    float rim = pow(1.0 - cosTheta, max(uRimPower, EPSILON));

    vec3 color = mix(volumeColor * uCrystalTint, reflectionColor, fresnel);
    color += uSunColor * specular * uSpecularStrength;
    color += uCrystalTint * rim * uRimStrength;
    return color;
}

// fix final render from above to make it look pretty
vec3 postProcess(vec3 color, vec2 uv01)
{
    color = max(color, vec3(0.0));
    color = 1.0 - exp(-color * max(uToneExposure, 0.0));
    color = pow(color, vec3(1.0 / max(uGamma, EPSILON)));

    float vignetteShape = max(uVignetteScale * uv01.x * uv01.y * (1.0 - uv01.x) * (1.0 - uv01.y), 0.0);
    float vignette = uVignetteMin + (1.0 - uVignetteMin) * pow(vignetteShape, max(uVignettePower, EPSILON));

    return color * vignette;
}

// get res, get pix coord, fix camera, shoot ray, do magic
void main()
{
    vec2 resolution = max(uResolution, vec2(1.0));
    vec2 uv01 = gl_FragCoord.xy / resolution; // pixel pos 0-1
    vec2 p = uv01 * 2.0 - 1.0;
    p.x *= resolution.x / resolution.y;

    vec3 ro = uCameraBaseScale * uCameraBase + uCameraOffset; // ray origin
    ro.xz *= rotation(uCameraYawBase + uCameraYawAmplitude * sin(uTime * uCameraYawSpeed));
    ro.yz *= rotation(uCameraPitchBase + uCameraPitchAmplitude * sin(uTime * uCameraPitchSpeed));

    vec3 forward = normalize(uCameraTarget - ro); // camera forward
    vec3 worldUp = abs(forward.y) > 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(worldUp, forward)); // camera right
    vec3 up = normalize(cross(forward, right)); // camera up
    vec3 rd = normalize(p.x * right + p.y * up + max(uFocalLength, EPSILON) * forward); // ray direction

    vec3 color = render(ro, rd); // final magic
    fragColor = vec4(postProcess(color, uv01), 1.0);
}
