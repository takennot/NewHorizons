set_project("NewHorizons")
set_version("7.0.0")

set_languages("c++20")

add_rules("mode.debug", "mode.release")

add_requires("glfw")

add_requires("glbinding")

add_requires("imgui", {
    configs = {
        glfw = true,
        opengl3 = true
    }
})



target("NewHorizons")

    set_kind("binary")

    add_files("src/*.cpp")

    add_packages("glfw", "glbinding", "imgui")

    add_defines("GLFW_INCLUDE_NONE")

    if is_plat("windows") then
        -- These two defines keep Windows headers smaller and stop the Windows
        -- min/max macros from colliding with std::min/std::max.
        add_defines("NOMINMAX", "WIN32_LEAN_AND_MEAN")
    end

    -- Copy the frag shader next
    -- to the executable after each build so the program can load it at runtime.
    after_build(function (target)
        local shaderdir = path.join(target:targetdir(), "shaders")
        os.mkdir(shaderdir)
        os.cp(path.join(os.projectdir(), "shaders", "*.frag"), shaderdir)
    end)
