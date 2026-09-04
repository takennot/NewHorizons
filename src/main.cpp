#include <GLFW/glfw3.h>
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace gl;

namespace
{
constexpr float PI = 3.14159265358979323846f;

// settings for imgui
// straight up asked coach which settings should I have
struct RenderSettings
{
    // Crystal
    float crystalRadius = 0.5f;
    float refractionRatio = 0.85f;

    // Sky + light
    float lightPos[3] = {3.0f, 4.0f, 2.0f};
    float skyLow[3] = {0.015f, 0.020f, 0.065f};
    float skyHigh[3] = {0.080f, 0.015f, 0.160f};
    float sunColor[3] = {1.00f, 0.58f, 0.92f};
    float skyHorizonScale = 0.5f;
    float skyHorizonBias = 0.5f;
    float sunGlowStrength = 0.35f;
    float sunGlowPower = 24.0f;
    float sunCoreStrength = 3.0f;
    float sunCorePower = 420.0f;

    // Fractal core
    float fractalOffset[3] = {-0.5f, -0.4f, -1.487f};
    float fractalOffsetScale = 1.01f;
    float fieldSeedDivisor = 4.0f;
    float fieldWeightDecay = 5.0f;
    float fieldResponseExponent = 2.2f;
    float fieldContrast = 5.2f;
    float fieldDensityBias = 0.65f;
    float hashScale = 0.1031f;
    float hashOffset = 33.33f;

    // Nebula motion/domains
    float driftSpeed[3] = {1.0f / 32.0f, 1.0f / 24.0f, 1.0f / 64.0f};
    float domain1Scale = 1.55f;
    float domain1Offset[3] = {0.8f, -1.3f, 0.0f};
    float domain1CameraInfluence = 0.35f;
    float domain1DriftInfluence = 0.20f;
    float domain2Scale = 1.5f;
    float domain2ScaleOscAmplitude = 0.12f;
    float domain2ScaleOscSpeed = 0.11f;
    float domain2Offset[3] = {2.0f, -1.3f, -1.0f};
    float domain2CameraInfluence = 0.35f;
    float domain2DriftInfluence = 0.10f;

    // Fractal layers
    float fieldSeed1 = 0.15f;
    int fieldIterations1 = 13;
    float fieldStrength1 = 9.025f;
    float fieldSeed2 = 0.90f;
    int fieldIterations2 = 18;
    float fieldStrength2 = 9.025f;
    float layer1Weights[3] = {0.12f, 0.85f, 2.40f};
    float layer1Powers[3] = {3.0f, 2.0f, 1.0f};
    float layer2Weights[3] = {3.80f, 0.32f, 1.65f};
    float layer2Powers[3] = {3.0f, 2.0f, 1.0f};

    // Volume
    int volumeSteps = 36;
    float volumeSampleOffset = 0.5f;
    float innerMaskStart = 0.50f;
    float innerMaskEnd = 0.97f;
    float nebulaScale = 2.0f;
    float densityWeights[3] = {0.18f, 0.34f, 0.48f};
    float densityScale = 0.32f;
    float nebulaEmissionStrength = 0.15f;
    float absorptionDensityScale = 2.2f;
    float absorptionBase = 0.03f;
    float volumeEmissionGain = 8.0f;
    float transmittanceCutoff = 0.015f;
    float deepTint[3] = {0.040f, 0.006f, 0.085f};
    float deepTintStrength = 1.0f;

    // Stars
    float starGridScale = 350.0f;
    float starThreshold = 0.9985f;
    float starCameraInfluence = 0.04f;
    float starColor[3] = {0.85f, 0.95f, 1.25f};
    float starIntensity = 1.55f;

    // Surface/glass
    float fresnelBase = 0.025f;
    float fresnelScale = 0.72f;
    float fresnelPower = 4.0f;
    float surfaceBias = 0.0015f;
    float exitSkyStrength = 0.11f;
    float specularPower = 2800.0f;
    float specularStrength = 1.6f;
    float rimPower = 15.0f;
    float rimStrength = 0.08f;
    float crystalTint[3] = {0.72f, 0.90f, 1.30f};

    // Procedural-space offset controlled by WASD/Space/Ctrl
    float proceduralPosition[3] = {0.0f, 0.0f, 0.0f};

    // Actual orbit camera
    float cameraBase[3] = {2.0f, 0.0f, 0.2f};
    float cameraBaseScale = 0.45f;
    float cameraOffset[3] = {0.0f, 0.75f, 0.0f};
    float cameraYawBase = PI * 0.5f;
    float cameraYawAmplitude = 1.0f;
    float cameraYawSpeed = 0.15f;
    float cameraPitchBase = 0.5f;
    float cameraPitchAmplitude = 0.125f;
    float cameraPitchSpeed = 0.15f * 0.70710678118f;
    float cameraTarget[3] = {0.0f, 0.0f, 0.0f};
    float focalLength = 2.0f;

    // Post-processing
    float toneExposure = 1.35f;
    float gamma = 1.6f;
    float vignetteMin = 0.5f;
    float vignetteScale = 19.0f;
    float vignettePower = 0.7f;
};

struct RuntimeSettings
{
    bool paused = false;
    bool vsync = true;
    bool showControls = true;
    float timeScale = 1.0f;
    float movementSpeed = 0.65f;
};

// list of uniforms
#define SHADER_UNIFORMS(X) \
    X(resolution, uResolution) \
    X(time, uTime) \
    X(cameraPosition, uCameraPosition) \
    X(crystalRadius, uCrystalRadius) \
    X(refractionRatio, uRefractionRatio) \
    X(lightPos, uLightPos) \
    X(skyLow, uSkyLow) \
    X(skyHigh, uSkyHigh) \
    X(sunColor, uSunColor) \
    X(skyHorizonScale, uSkyHorizonScale) \
    X(skyHorizonBias, uSkyHorizonBias) \
    X(sunGlowStrength, uSunGlowStrength) \
    X(sunGlowPower, uSunGlowPower) \
    X(sunCoreStrength, uSunCoreStrength) \
    X(sunCorePower, uSunCorePower) \
    X(fractalOffset, uFractalOffset) \
    X(fractalOffsetScale, uFractalOffsetScale) \
    X(fieldSeedDivisor, uFieldSeedDivisor) \
    X(fieldWeightDecay, uFieldWeightDecay) \
    X(fieldResponseExponent, uFieldResponseExponent) \
    X(fieldContrast, uFieldContrast) \
    X(fieldDensityBias, uFieldDensityBias) \
    X(driftSpeed, uDriftSpeed) \
    X(domain1Scale, uDomain1Scale) \
    X(domain1Offset, uDomain1Offset) \
    X(domain1CameraInfluence, uDomain1CameraInfluence) \
    X(domain1DriftInfluence, uDomain1DriftInfluence) \
    X(domain2Scale, uDomain2Scale) \
    X(domain2ScaleOscAmplitude, uDomain2ScaleOscAmplitude) \
    X(domain2ScaleOscSpeed, uDomain2ScaleOscSpeed) \
    X(domain2Offset, uDomain2Offset) \
    X(domain2CameraInfluence, uDomain2CameraInfluence) \
    X(domain2DriftInfluence, uDomain2DriftInfluence) \
    X(fieldSeed1, uFieldSeed1) \
    X(fieldIterations1, uFieldIterations1) \
    X(fieldStrength1, uFieldStrength1) \
    X(fieldSeed2, uFieldSeed2) \
    X(fieldIterations2, uFieldIterations2) \
    X(fieldStrength2, uFieldStrength2) \
    X(layer1Weights, uLayer1Weights) \
    X(layer1Powers, uLayer1Powers) \
    X(layer2Weights, uLayer2Weights) \
    X(layer2Powers, uLayer2Powers) \
    X(hashScale, uHashScale) \
    X(hashOffset, uHashOffset) \
    X(volumeSteps, uVolumeSteps) \
    X(volumeSampleOffset, uVolumeSampleOffset) \
    X(innerMaskStart, uInnerMaskStart) \
    X(innerMaskEnd, uInnerMaskEnd) \
    X(nebulaScale, uNebulaScale) \
    X(densityWeights, uDensityWeights) \
    X(densityScale, uDensityScale) \
    X(starGridScale, uStarGridScale) \
    X(starThreshold, uStarThreshold) \
    X(starCameraInfluence, uStarCameraInfluence) \
    X(starColor, uStarColor) \
    X(starIntensity, uStarIntensity) \
    X(nebulaEmissionStrength, uNebulaEmissionStrength) \
    X(absorptionDensityScale, uAbsorptionDensityScale) \
    X(absorptionBase, uAbsorptionBase) \
    X(volumeEmissionGain, uVolumeEmissionGain) \
    X(transmittanceCutoff, uTransmittanceCutoff) \
    X(deepTint, uDeepTint) \
    X(deepTintStrength, uDeepTintStrength) \
    X(fresnelBase, uFresnelBase) \
    X(fresnelScale, uFresnelScale) \
    X(fresnelPower, uFresnelPower) \
    X(surfaceBias, uSurfaceBias) \
    X(exitSkyStrength, uExitSkyStrength) \
    X(specularPower, uSpecularPower) \
    X(specularStrength, uSpecularStrength) \
    X(rimPower, uRimPower) \
    X(rimStrength, uRimStrength) \
    X(crystalTint, uCrystalTint) \
    X(cameraBase, uCameraBase) \
    X(cameraBaseScale, uCameraBaseScale) \
    X(cameraOffset, uCameraOffset) \
    X(cameraYawBase, uCameraYawBase) \
    X(cameraYawAmplitude, uCameraYawAmplitude) \
    X(cameraYawSpeed, uCameraYawSpeed) \
    X(cameraPitchBase, uCameraPitchBase) \
    X(cameraPitchAmplitude, uCameraPitchAmplitude) \
    X(cameraPitchSpeed, uCameraPitchSpeed) \
    X(cameraTarget, uCameraTarget) \
    X(focalLength, uFocalLength) \
    X(toneExposure, uToneExposure) \
    X(gamma, uGamma) \
    X(vignetteMin, uVignetteMin) \
    X(vignetteScale, uVignetteScale) \
    X(vignettePower, uVignettePower)

// get uniforms using that abomination above
struct UniformLocations
{
    // set all SHADER_UNIFORMS to -1
    #define DECLARE_UNIFORM(field, glslName) GLint field = -1;
        SHADER_UNIFORMS(DECLARE_UNIFORM)
    #undef DECLARE_UNIFORM

    void query(GLuint program)
    {
        // get uniform location but for all
        #define QUERY_UNIFORM(field, glslName) field = glGetUniformLocation(program, #glslName);
                SHADER_UNIFORMS(QUERY_UNIFORM)
        #undef QUERY_UNIFORM
    }

    void validateRequired() const
    {
        if (resolution < 0 || time < 0 || cameraPosition < 0)
        {
            throw std::runtime_error("Expected uResolution, uTime, and uCameraPosition uniforms were not found.");
        }
    }
};

// get shader file
std::string readTextFile(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// hardcodin shader name for now, maybe can make a drop down list later in imgui
fs::path findShaderPath(const char* argv0)
{
    const fs::path relative = fs::path("shaders") / "crystal.frag";

    if (fs::exists(relative))
    {
        return fs::absolute(relative);
    }

    std::error_code error;
    const fs::path executable = fs::absolute(argv0, error);
    if (!error)
    {
        const fs::path besideExecutable = executable.parent_path() / relative;
        if (fs::exists(besideExecutable))
        {
            return besideExecutable;
        }
    }

    throw std::runtime_error(
        "Could not locate shaders/crystal.frag. Keep the shaders directory in the project "
        "or beside the xmake-built executable.");
}

// you are not gonna believe this - compiling shader here :O
GLuint compileShader(GLenum type, const std::string& source, const char* label)
{
    const GLuint shader = glCreateShader(type);
    const char* sourcePtr = source.c_str();
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);

    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != 0)
    {
        return shader;
    }

    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error(std::string("Shader compile failed (") + label + "):\n" + log);
}

GLuint createProgram(const std::string& fragmentSource)
{
    // its too small to be its own file - let it be here lol
    static constexpr const char* vertexSource = R"GLSL(
        #version 330 core
        const vec2 positions[3] = vec2[3](
            vec2(-1.0, -1.0),
            vec2( 3.0, -1.0),
            vec2(-1.0,  3.0)
        );
        void main()
        {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
        )GLSL";

    GLuint vertexShader = 0;
    GLuint fragmentShader = 0;
    GLuint program = 0;

    try
    {
        vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource, "fullscreen vertex shader");
        fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource, "crystal.frag");
        program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        GLint linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (linked == 0)
        {
            GLint length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
            std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
            glGetProgramInfoLog(program, length, nullptr, log.data());
            throw std::runtime_error("Program link failed:\n" + log);
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return program;
    }
    catch (...)
    {
        if (vertexShader != 0)
        {
            glDeleteShader(vertexShader);
        }

        if (fragmentShader != 0)
        {
            glDeleteShader(fragmentShader);
        }

        if (program != 0)
        {
            glDeleteProgram(program);
        }
        throw;
    }
}

void glfwErrorCallback(int error, const char* description)
{
    std::cerr << "GLFW error " << error << ": " << description << '\n';
}

// send data to shader
void uploadUniforms(const UniformLocations& u, const RenderSettings& s, int width, int height, float shaderTime)
{
    #define SET1F(location, value) do { if ((location) >= 0) glUniform1f((location), (value)); } while (false)
    #define SET1I(location, value) do { if ((location) >= 0) glUniform1i((location), (value)); } while (false)
    #define SET3F(location, value) do { if ((location) >= 0) glUniform3fv((location), 1, (value)); } while (false)

    if (u.resolution >= 0)
    {
        glUniform2f(u.resolution, static_cast<float>(width), static_cast<float>(height));
    }

    SET1F(u.time, shaderTime);
    SET3F(u.cameraPosition, s.proceduralPosition);

    SET1F(u.crystalRadius, s.crystalRadius);
    SET1F(u.refractionRatio, s.refractionRatio);
    SET3F(u.lightPos, s.lightPos);
    SET3F(u.skyLow, s.skyLow);
    SET3F(u.skyHigh, s.skyHigh);
    SET3F(u.sunColor, s.sunColor);
    SET1F(u.skyHorizonScale, s.skyHorizonScale);
    SET1F(u.skyHorizonBias, s.skyHorizonBias);
    SET1F(u.sunGlowStrength, s.sunGlowStrength);
    SET1F(u.sunGlowPower, s.sunGlowPower);
    SET1F(u.sunCoreStrength, s.sunCoreStrength);
    SET1F(u.sunCorePower, s.sunCorePower);

    SET3F(u.fractalOffset, s.fractalOffset);
    SET1F(u.fractalOffsetScale, s.fractalOffsetScale);
    SET1F(u.fieldSeedDivisor, s.fieldSeedDivisor);
    SET1F(u.fieldWeightDecay, s.fieldWeightDecay);
    SET1F(u.fieldResponseExponent, s.fieldResponseExponent);
    SET1F(u.fieldContrast, s.fieldContrast);
    SET1F(u.fieldDensityBias, s.fieldDensityBias);
    SET1F(u.hashScale, s.hashScale);
    SET1F(u.hashOffset, s.hashOffset);

    SET3F(u.driftSpeed, s.driftSpeed);
    SET1F(u.domain1Scale, s.domain1Scale);
    SET3F(u.domain1Offset, s.domain1Offset);
    SET1F(u.domain1CameraInfluence, s.domain1CameraInfluence);
    SET1F(u.domain1DriftInfluence, s.domain1DriftInfluence);
    SET1F(u.domain2Scale, s.domain2Scale);
    SET1F(u.domain2ScaleOscAmplitude, s.domain2ScaleOscAmplitude);
    SET1F(u.domain2ScaleOscSpeed, s.domain2ScaleOscSpeed);
    SET3F(u.domain2Offset, s.domain2Offset);
    SET1F(u.domain2CameraInfluence, s.domain2CameraInfluence);
    SET1F(u.domain2DriftInfluence, s.domain2DriftInfluence);

    SET1F(u.fieldSeed1, s.fieldSeed1);
    SET1I(u.fieldIterations1, s.fieldIterations1);
    SET1F(u.fieldStrength1, s.fieldStrength1);
    SET1F(u.fieldSeed2, s.fieldSeed2);
    SET1I(u.fieldIterations2, s.fieldIterations2);
    SET1F(u.fieldStrength2, s.fieldStrength2);
    SET3F(u.layer1Weights, s.layer1Weights);
    SET3F(u.layer1Powers, s.layer1Powers);
    SET3F(u.layer2Weights, s.layer2Weights);
    SET3F(u.layer2Powers, s.layer2Powers);

    SET1I(u.volumeSteps, s.volumeSteps);
    SET1F(u.volumeSampleOffset, s.volumeSampleOffset);
    SET1F(u.innerMaskStart, s.innerMaskStart);
    SET1F(u.innerMaskEnd, s.innerMaskEnd);
    SET1F(u.nebulaScale, s.nebulaScale);
    SET3F(u.densityWeights, s.densityWeights);
    SET1F(u.densityScale, s.densityScale);
    SET1F(u.nebulaEmissionStrength, s.nebulaEmissionStrength);
    SET1F(u.absorptionDensityScale, s.absorptionDensityScale);
    SET1F(u.absorptionBase, s.absorptionBase);
    SET1F(u.volumeEmissionGain, s.volumeEmissionGain);
    SET1F(u.transmittanceCutoff, s.transmittanceCutoff);
    SET3F(u.deepTint, s.deepTint);
    SET1F(u.deepTintStrength, s.deepTintStrength);

    SET1F(u.starGridScale, s.starGridScale);
    SET1F(u.starThreshold, s.starThreshold);
    SET1F(u.starCameraInfluence, s.starCameraInfluence);
    SET3F(u.starColor, s.starColor);
    SET1F(u.starIntensity, s.starIntensity);

    SET1F(u.fresnelBase, s.fresnelBase);
    SET1F(u.fresnelScale, s.fresnelScale);
    SET1F(u.fresnelPower, s.fresnelPower);
    SET1F(u.surfaceBias, s.surfaceBias);
    SET1F(u.exitSkyStrength, s.exitSkyStrength);
    SET1F(u.specularPower, s.specularPower);
    SET1F(u.specularStrength, s.specularStrength);
    SET1F(u.rimPower, s.rimPower);
    SET1F(u.rimStrength, s.rimStrength);
    SET3F(u.crystalTint, s.crystalTint);

    SET3F(u.cameraBase, s.cameraBase);
    SET1F(u.cameraBaseScale, s.cameraBaseScale);
    SET3F(u.cameraOffset, s.cameraOffset);
    SET1F(u.cameraYawBase, s.cameraYawBase);
    SET1F(u.cameraYawAmplitude, s.cameraYawAmplitude);
    SET1F(u.cameraYawSpeed, s.cameraYawSpeed);
    SET1F(u.cameraPitchBase, s.cameraPitchBase);
    SET1F(u.cameraPitchAmplitude, s.cameraPitchAmplitude);
    SET1F(u.cameraPitchSpeed, s.cameraPitchSpeed);
    SET3F(u.cameraTarget, s.cameraTarget);
    SET1F(u.focalLength, s.focalLength);

    SET1F(u.toneExposure, s.toneExposure);
    SET1F(u.gamma, s.gamma);
    SET1F(u.vignetteMin, s.vignetteMin);
    SET1F(u.vignetteScale, s.vignetteScale);
    SET1F(u.vignettePower, s.vignettePower);

    #undef SET1F
    #undef SET1I
    #undef SET3F
}

// color picker
bool hdrColor(const char* label, float color[3])
{
    return ImGui::ColorEdit3(label, color, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
}

static void settingTooltip(const char* description)
{
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::SetTooltip("%s", description);
    }
}

struct UiActions
{
    bool reloadShader = false;
    bool resetTime = false;
    bool vsyncChanged = false;
};

// HOLY IMGUI
UiActions drawControls(RenderSettings& s, RuntimeSettings& runtime, float shaderTime, const std::string& shaderStatus)
{
    UiActions actions;

    ImGui::SetNextWindowSize(ImVec2(470.0f, 860.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Settings", &runtime.showControls))
    {
        ImGui::End();
        return actions;
    }

    ImGui::Text("%.1f FPS  |  %.2f ms", ImGui::GetIO().Framerate, 1000.0f / std::max(ImGui::GetIO().Framerate, 0.001f));
    ImGui::TextWrapped("%s", shaderStatus.c_str());

    // technically speaking why do I even have that if Im literally ediitng the shader with imgui? xD
    if (ImGui::Button("Reload shader (R)"))
    {
        actions.reloadShader = true;
    }

    ImGui::SameLine();

    // resets all render settings but not runtime settings
    if (ImGui::Button("Reset all"))
    {
        s = RenderSettings{};
    }

    ImGui::SameLine();

    if (ImGui::Button("Reset time"))
    {
        actions.resetTime = true;
    }

    if (ImGui::CollapsingHeader("Runtime", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Pause animation", &runtime.paused);
        settingTooltip("Really? You need a tooltip for that?");

        if (ImGui::Checkbox("VSync", &runtime.vsync))
        {
            actions.vsyncChanged = true;
        }
        settingTooltip("You don't need a tooltip for that.");

        ImGui::DragFloat("Time scale", &runtime.timeScale, 0.01f, -5.0f, 5.0f, "%.3f");
        settingTooltip("Controls how fast everything animates. Higher = faster. Negative = slower / animation runs backwards.");

        ImGui::DragFloat("WASD movement speed", &runtime.movementSpeed, 0.01f, 0.0f, 10.0f, "%.3f");
        settingTooltip("Controls how fast insides of the crystal move with WASD, Space and CTRL.");

        ImGui::TextUnformatted("F1 toggles this panel. WASD/Space/CTRL moves procedural space.");
    }

    if (ImGui::CollapsingHeader("Crystal", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat("Radius", &s.crystalRadius, 0.002f, 0.01f, 5.0f, "%.4f");
        settingTooltip("Crystal size.");

        ImGui::DragFloat("Refraction ratio", &s.refractionRatio, 0.001f, 0.05f, 2.0f, "%.4f");
        settingTooltip("Controls how strongly the crystal bends light passing through it. Bigger changes make the inside look more distorted.");
    }

    if (ImGui::CollapsingHeader("Sky + light", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat3("Light position", s.lightPos, 0.02f, -50.0f, 50.0f, "%.3f");
        settingTooltip("Moves the light source around the crystal. This changes where highlights appear.");

        hdrColor("Sky low", s.skyLow);
        settingTooltip("Changes the color of the sky near the bottom and horizon.");

        hdrColor("Sky high", s.skyHigh);
        settingTooltip("Changes the color of the sky higher up.");

        hdrColor("Sun color", s.sunColor);
        settingTooltip("Changes the color of the sun and its light.");

        ImGui::DragFloat("Horizon scale", &s.skyHorizonScale, 0.005f, -4.0f, 4.0f, "%.3f");
        settingTooltip("Controls how quickly the sky changes from the lower color to the upper color.");

        ImGui::DragFloat("Horizon bias", &s.skyHorizonBias, 0.005f, -2.0f, 2.0f, "%.3f");
        settingTooltip("Moves the sky's color transition higher or lower.");

        ImGui::DragFloat("Sun glow strength", &s.sunGlowStrength, 0.01f, 0.0f, 20.0f, "%.3f");
        settingTooltip("Controls how bright the large soft glow around the sun is.");

        ImGui::DragFloat("Sun glow power", &s.sunGlowPower, 0.2f, 0.1f, 1000.0f, "%.2f");
        settingTooltip("Controls the size of the sun's soft glow. Higher values make the glow tighter and smaller.");

        ImGui::DragFloat("Sun core strength", &s.sunCoreStrength, 0.02f, 0.0f, 50.0f, "%.3f");
        settingTooltip("Controls how bright the center of the sun is.");

        ImGui::DragFloat("Sun core power", &s.sunCorePower, 1.0f, 0.1f, 5000.0f, "%.1f");
        settingTooltip("Controls how small and sharp the bright center of the sun is. Higher = smaller and sharper.");
    }

    if (ImGui::CollapsingHeader("Fractal field", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat3("Fractal offset", s.fractalOffset, 0.002f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Changes the shape and pattern of the nebula fractal inside the crystal.");

        ImGui::DragFloat("Offset scale", &s.fractalOffsetScale, 0.002f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how strongly the fractal offset changes the nebula pattern.");

        ImGui::DragFloat("Seed divisor", &s.fieldSeedDivisor, 0.01f, 0.001f, 50.0f, "%.3f");
        settingTooltip("Changes how much the seed affects the generated pattern.");

        ImGui::DragFloat("Weight decay", &s.fieldWeightDecay, 0.02f, 0.01f, 50.0f, "%.3f");
        settingTooltip("Controls how much fine fractal detail is kept. Higher values keep more layers of detail.");

        ImGui::DragFloat("Response exponent", &s.fieldResponseExponent, 0.01f, 0.01f, 10.0f, "%.3f");
        settingTooltip("Changes how sharply the fractal reacts to differences in its shape. Higher values can make features more pronounced.");

        ImGui::DragFloat("Contrast", &s.fieldContrast, 0.02f, 0.0f, 30.0f, "%.3f");
        settingTooltip("Makes the internal pattern stronger or weaker.");

        ImGui::DragFloat("Density bias", &s.fieldDensityBias, 0.005f, -5.0f, 10.0f, "%.3f");
        settingTooltip("Controls how much of the material appears. Higher values remove weaker areas and leave fewer clouds.");

        ImGui::DragFloat("Hash scale", &s.hashScale, 0.0001f, -5.0f, 5.0f, "%.5f");
        settingTooltip("Changes the random pattern used to place stars. Adjusting it gives the stars a different arrangement.");

        ImGui::DragFloat("Hash offset", &s.hashOffset, 0.02f, -100.0f, 100.0f, "%.3f");
        settingTooltip("Also changes the random star pattern. Use it to get a different star arrangement.");
    }

    if (ImGui::CollapsingHeader("Nebula domains + animation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat3("Drift speed XYZ", s.driftSpeed, 0.0005f, -2.0f, 2.0f, "%.5f");
        settingTooltip("Controls how fast the nebula moves and changes in each direction.");

        ImGui::SeparatorText("Domain 1");

        ImGui::DragFloat("D1 Scale", &s.domain1Scale, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Changes the size of the first cloud pattern. Larger values usually create smaller, more frequent details.");

        ImGui::DragFloat3("D1 Offset", s.domain1Offset, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Moves the first cloud pattern around without moving the crystal.");

        ImGui::DragFloat("D1 Camera influence", &s.domain1CameraInfluence, 0.005f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how much moving through procedural space shifts the first cloud pattern.");

        ImGui::DragFloat("D1 Drift influence", &s.domain1DriftInfluence, 0.005f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how strongly the first cloud pattern is affected by the animation.");

        ImGui::DragFloat("D1 Seed", &s.fieldSeed1, 0.005f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Changes the starting value used to create the first cloud pattern.");

        ImGui::SliderInt("D1 Iterations", &s.fieldIterations1, 1, 64);
        settingTooltip("Controls how many times the first fractal is calculated. More can create more detail but costs more performance.");

        ImGui::DragFloat("D1 Strength", &s.fieldStrength1, 0.02f, 0.0f, 100.0f, "%.3f");
        settingTooltip("Changes the shape and sharpness of the first cloud layer.");

        ImGui::DragFloat3("D1 RGB weights", s.layer1Weights, 0.01f, -20.0f, 20.0f, "%.3f");
        settingTooltip("Controls how much red, green and blue the first cloud layer contributes.");

        ImGui::DragFloat3("D1 RGB powers", s.layer1Powers, 0.01f, 0.01f, 10.0f, "%.3f");
        settingTooltip("Changes how strongly bright and dark parts of the first cloud layer affect each color.");

        ImGui::SeparatorText("Domain 2");

        ImGui::DragFloat("D2 Base scale", &s.domain2Scale, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Changes the size of the second cloud pattern. Larger absolute values usually create smaller, more frequent details.");

        ImGui::DragFloat("D2 Scale oscillation amplitude", &s.domain2ScaleOscAmplitude, 0.002f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Controls how much the second cloud pattern grows and shrinks over time. Zero disables this size animation.");

        ImGui::DragFloat("D2 Scale oscillation speed", &s.domain2ScaleOscSpeed, 0.002f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Controls how fast the second cloud pattern grows and shrinks.");

        ImGui::DragFloat3("D2 Offset", s.domain2Offset, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Moves the second cloud pattern around without moving the crystal.");

        ImGui::DragFloat("D2 Camera influence", &s.domain2CameraInfluence, 0.005f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how much moving through procedural space shifts the second cloud pattern.");

        ImGui::DragFloat("D2 Drift influence", &s.domain2DriftInfluence, 0.005f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how strongly the second cloud pattern is affected by the animation.");

        ImGui::DragFloat("D2 Seed", &s.fieldSeed2, 0.005f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Changes the starting value used to create the second cloud pattern. Useful for finding different looks.");

        ImGui::SliderInt("D2 Iterations", &s.fieldIterations2, 1, 64);
        settingTooltip("Controls how many times the second fractal is calculated. More can create more detail but costs more performance.");

        ImGui::DragFloat("D2 Strength", &s.fieldStrength2, 0.02f, 0.0f, 100.0f, "%.3f");
        settingTooltip("Changes the shape and sharpness of the second cloud layer.");

        ImGui::DragFloat3("D2 RGB weights", s.layer2Weights, 0.01f, -20.0f, 20.0f, "%.3f");
        settingTooltip("Controls how much red, green and blue the second cloud layer contributes.");

        ImGui::DragFloat3("D2 RGB powers", s.layer2Powers, 0.01f, 0.01f, 10.0f, "%.3f");
        settingTooltip("Changes how strongly bright and dark parts of the second cloud layer affect each color.");
    }

    if (ImGui::CollapsingHeader("Volume rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("Raymarch steps", &s.volumeSteps, 1, 128);
        settingTooltip("Controls how many samples are taken through the inside of the crystal. More = smoother and more accurate, but slower.");

        ImGui::SliderFloat("Sample offset", &s.volumeSampleOffset, 0.0f, 1.0f, "%.3f");
        settingTooltip("Moves where each raymarch sample is taken. Usually causes small changes in the appearance of fine details.");

        ImGui::SliderFloat("Inner mask start", &s.innerMaskStart, 0.0f, 1.5f, "%.4f");
        settingTooltip("Controls where the nebula begins fading away as it approaches the outside of the crystal.");

        ImGui::SliderFloat("Inner mask end", &s.innerMaskEnd, 0.0f, 1.5f, "%.4f");
        settingTooltip("Controls where the nebula finishes fading away near the outside of the crystal.");

        ImGui::DragFloat("Nebula coordinate scale", &s.nebulaScale, 0.01f, -20.0f, 20.0f, "%.3f");
        settingTooltip("Changes the overall size of the nebula pattern. Larger values usually make its details smaller.");

        ImGui::DragFloat3("Density RGB weights", s.densityWeights, 0.005f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how much the red, green and blue parts of the nebula contribute to its thickness.");

        ImGui::DragFloat("Density scale", &s.densityScale, 0.005f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Controls how thick or dense the nebula is.");

        ImGui::DragFloat("Nebula emission", &s.nebulaEmissionStrength, 0.005f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Controls how strongly the nebula itself glows.");

        ImGui::DragFloat("Absorption density scale", &s.absorptionDensityScale, 0.01f, 0.0f, 50.0f, "%.3f");
        settingTooltip("Controls how strongly dense clouds block light. Higher values make thick areas more opaque.");

        ImGui::DragFloat("Base absorption", &s.absorptionBase, 0.001f, 0.0f, 10.0f, "%.4f");
        settingTooltip("Adds general light blocking throughout the inside of the crystal, even where the clouds are weak.");

        ImGui::DragFloat("Emission gain", &s.volumeEmissionGain, 0.05f, -50.0f, 100.0f, "%.3f");
        settingTooltip("Controls the overall brightness of glowing material and stars inside the crystal.");

        ImGui::DragFloat("Early-exit transmittance", &s.transmittanceCutoff, 0.0005f, 0.0f, 1.0f, "%.4f");
        settingTooltip("Performance setting. Stops calculating a ray once almost no light can pass through it. Higher values stop sooner but may lose detail.");

        hdrColor("Deep tint", s.deepTint);
        settingTooltip("Changes the color added to thick, deep areas inside the crystal.");

        ImGui::DragFloat("Deep tint strength", &s.deepTintStrength, 0.01f, -10.0f, 20.0f, "%.3f");
        settingTooltip("Controls how strongly the deep tint color appears.");
    }

    if (ImGui::CollapsingHeader("Stars", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat("Grid scale", &s.starGridScale, 1.0f, 1.0f, 5000.0f, "%.1f");
        settingTooltip("Changes how tightly the possible star positions are packed. Higher values create a finer star grid.");

        ImGui::SliderFloat("Spawn threshold", &s.starThreshold, 0.90f, 1.0f, "%.6f");
        settingTooltip("Controls how rare stars are. Higher = fewer stars. Lower = more stars.");

        ImGui::DragFloat("Camera influence", &s.starCameraInfluence, 0.001f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how much moving through procedural space changes the star pattern.");

        hdrColor("Star color", s.starColor);
        settingTooltip("Changes the color of the stars inside the crystal.");

        ImGui::DragFloat("Star intensity", &s.starIntensity, 0.01f, 0.0f, 100.0f, "%.3f");
        settingTooltip("Controls how brightly the stars glow.");
    }

    if (ImGui::CollapsingHeader("Glass surface", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat("Fresnel base", &s.fresnelBase, 0.001f, -2.0f, 2.0f, "%.4f");
        settingTooltip("Controls the minimum amount of sky reflection visible on the crystal.");

        ImGui::DragFloat("Fresnel scale", &s.fresnelScale, 0.005f, -5.0f, 5.0f, "%.4f");
        settingTooltip("Controls how much stronger the reflection becomes around the edges of the crystal.");

        ImGui::DragFloat("Fresnel power", &s.fresnelPower, 0.02f, 0.01f, 100.0f, "%.3f");
        settingTooltip("Controls how tightly the edge reflection is concentrated around the outside of the crystal.");

        // so that 2 rays don't render same thing & renderer doesn't start crashing out
        ImGui::DragFloat("Surface bias", &s.surfaceBias, 0.00001f, 0.000001f, 0.1f, "%.6f");
        settingTooltip("Offset used to prevent rendering errors where rays enter the crystal. Do NOT change this.");

        ImGui::DragFloat("Exit sky strength", &s.exitSkyStrength, 0.005f, 0.0f, 10.0f, "%.4f");
        settingTooltip("Controls how much of the outside sky can be seen through the crystal.");

        ImGui::DragFloat("Specular power", &s.specularPower, 5.0f, 0.01f, 20000.0f, "%.1f");
        settingTooltip("Controls the size of the shiny light spot on the crystal. Higher = smaller and sharper.");

        ImGui::DragFloat("Specular strength", &s.specularStrength, 0.01f, 0.0f, 100.0f, "%.3f");
        settingTooltip("Controls how bright the shiny light spot is.");

        ImGui::DragFloat("Rim power", &s.rimPower, 0.05f, 0.01f, 200.0f, "%.2f");
        settingTooltip("Controls how thin or wide the glow around the edge of the crystal is. Higher = thinner.");

        ImGui::DragFloat("Rim strength", &s.rimStrength, 0.005f, 0.0f, 20.0f, "%.4f");
        settingTooltip("Controls how bright the edge glow is.");

        hdrColor("Crystal tint", s.crystalTint);
        settingTooltip("Changes the overall color tint of the crystal and its edge glow.");
    }

    if (ImGui::CollapsingHeader("Camera + procedural position", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat3("Procedural position", s.proceduralPosition, 0.01f, -100.0f, 100.0f, "%.3f");
        settingTooltip("Moves through the generated nebula and star world without moving the camera.");

        if (ImGui::Button("Reset procedural position")) {
            s.proceduralPosition[0] = s.proceduralPosition[1] = s.proceduralPosition[2] = 0.0f;
        }
        settingTooltip("Returns the nebula world position to its starting point.");

        ImGui::SeparatorText("Orbit camera");

        ImGui::DragFloat3("Camera base", s.cameraBase, 0.01f, -20.0f, 20.0f, "%.3f");
        settingTooltip("Sets the camera's basic position around the crystal.");

        ImGui::DragFloat("Camera base scale", &s.cameraBaseScale, 0.005f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Moves the camera closer to or farther from the crystal by scaling its base position.");

        ImGui::DragFloat3("Camera offset", s.cameraOffset, 0.01f, -20.0f, 20.0f, "%.3f");
        settingTooltip("Moves the camera away from its normal position.");

        ImGui::DragFloat("Yaw base", &s.cameraYawBase, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Sets the camera's starting rotation around the crystal from side to side.");

        ImGui::DragFloat("Yaw amplitude", &s.cameraYawAmplitude, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Controls how far the camera automatically swings from side to side. Zero disables this movement.");

        ImGui::DragFloat("Yaw speed", &s.cameraYawSpeed, 0.002f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Controls how fast the camera automatically swings from side to side.");

        ImGui::DragFloat("Pitch base", &s.cameraPitchBase, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Sets the camera's starting rotation above or below the crystal.");

        ImGui::DragFloat("Pitch amplitude", &s.cameraPitchAmplitude, 0.005f, -20.0f, 20.0f, "%.4f");
        settingTooltip("Controls how far the camera automatically swings up and down. Zero disables this movement.");

        ImGui::DragFloat("Pitch speed", &s.cameraPitchSpeed, 0.002f, -10.0f, 10.0f, "%.4f");
        settingTooltip("Controls how fast the camera automatically swings up and down.");

        ImGui::DragFloat3("Look target", s.cameraTarget, 0.01f, -20.0f, 20.0f, "%.3f");
        settingTooltip("Changes the point the camera looks at. Normally this is the center of the crystal.");

        ImGui::DragFloat("Focal length / zoom", &s.focalLength, 0.01f, 0.01f, 20.0f, "%.3f");
        settingTooltip("Controls the camera zoom. Higher values give a more zoomed-in, flatter view.");
    }

    if (ImGui::CollapsingHeader("Post-processing", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat("Exposure", &s.toneExposure, 0.01f, 0.0f, 20.0f, "%.3f");
        settingTooltip("Controls the overall brightness of the finished image. Higher = brighter.");

        ImGui::DragFloat("Gamma", &s.gamma, 0.01f, 0.05f, 10.0f, "%.3f");
        settingTooltip("Changes the brightness of the image, especially the darker and middle areas.");

        ImGui::SliderFloat("Vignette edge minimum", &s.vignetteMin, 0.0f, 1.0f, "%.3f");
        settingTooltip("Controls how dark the corners are allowed to become. Higher values keep the corners brighter.");

        ImGui::DragFloat("Vignette scale", &s.vignetteScale, 0.05f, 0.0f, 100.0f, "%.3f");
        settingTooltip("Controls the overall strength and brightness of the vignette effect.");

        ImGui::DragFloat("Vignette power", &s.vignettePower, 0.01f, 0.01f, 10.0f, "%.3f");
        settingTooltip("Changes the shape of the darkening around the edges of the image.");
    }

    ImGui::End();
    return actions;
}

bool pressedOnce(GLFWwindow* window, int key, bool& wasDown)
{
    const bool isDown = glfwGetKey(window, key) == GLFW_PRESS;
    const bool pressed = isDown && !wasDown;
    wasDown = isDown;
    return pressed;
}
} // namespace

int main(int argc, char** argv)
{
    (void)argc;
    glfwSetErrorCallback(glfwErrorCallback);

    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "Could not initialize GLFW.\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef _WIN32
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
    #endif

    // maybe add fullscreen switch later
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "New Horizons", glfwGetPrimaryMonitor(), nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glbinding::initialize([](const char* name) {
        return reinterpret_cast<glbinding::ProcAddress>(glfwGetProcAddress(name));
    });

    GLuint program = 0;
    GLuint vao = 0;
    bool imguiInitialized = false;

    try
    {
        const fs::path shaderPath = findShaderPath(argv[0]);
        program = createProgram(readTextFile(shaderPath));

        UniformLocations uniforms;
        uniforms.query(program);
        uniforms.validateRequired();

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330 core");
        imguiInitialized = true;

        RenderSettings settings;
        RuntimeSettings runtime;

        std::string reloadStatus = "Shader loaded.";
        auto reloadShader = [&]()
        {
            try
            {
                const GLuint newProgram = createProgram(readTextFile(shaderPath));
                UniformLocations newUniforms;
                newUniforms.query(newProgram);
                newUniforms.validateRequired();

                glDeleteProgram(program);
                program = newProgram;
                uniforms = newUniforms;
                reloadStatus = "Shader reloaded successfully.";
                std::cout << reloadStatus << '\n';
            }
            catch (const std::exception& error)
            {
                reloadStatus = std::string("Reload failed: ") + error.what();
                std::cerr << reloadStatus << '\n';
            }
        };

        bool reloadWasDown = false;
        bool panelWasDown = false;
        double previousTime = glfwGetTime();
        float shaderTime = static_cast<float>(previousTime);

        while (glfwWindowShouldClose(window) == GLFW_FALSE)
        {
            glfwPollEvents();

            const double currentTime = glfwGetTime();
            const float deltaTime = static_cast<float>(currentTime - previousTime);
            previousTime = currentTime;

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            if (pressedOnce(window, GLFW_KEY_F1, panelWasDown))
            {
                runtime.showControls = !runtime.showControls;
            }

            if (!runtime.paused)
            {
                shaderTime += deltaTime * runtime.timeScale;
            }

            UiActions actions;
            if (runtime.showControls)
            {
                actions = drawControls(settings, runtime, shaderTime, reloadStatus);
            }

            const bool reloadKey = pressedOnce(window, GLFW_KEY_R, reloadWasDown);
            if ((reloadKey && !io.WantCaptureKeyboard) || actions.reloadShader)
            {
                reloadShader();
            }

            if (actions.resetTime)
            {
                shaderTime = 0.0f;
            }

            if (actions.vsyncChanged)
            {
                glfwSwapInterval(runtime.vsync ? 1 : 0);
            }

            if (!io.WantCaptureKeyboard)
            {
                const float movement = runtime.movementSpeed * deltaTime;
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                    settings.proceduralPosition[0] -= movement;
                }

                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                    settings.proceduralPosition[0] += movement;
                }

                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                    settings.proceduralPosition[2] += movement;
                }

                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                    settings.proceduralPosition[2] -= movement;
                }

                if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                    settings.proceduralPosition[1] += movement;
                }

                if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
                    settings.proceduralPosition[1] -= movement;
                }
            }

            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            if (width <= 0 || height <= 0)
            {
                ImGui::Render();
                continue;
            }

            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(program);
            uploadUniforms(uniforms, settings, width, height, shaderTime);
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';

        if (imguiInitialized)
        {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
        if (vao != 0) {
            glDeleteVertexArrays(1, &vao);
        }

        if (program != 0) {
            glDeleteProgram(program);
        }

        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (imguiInitialized)
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
