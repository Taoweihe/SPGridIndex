set_languages("cxx20")
set_arch("x64")
set_kind("binary")
add_rules("mode.debug", "mode.release")
add_requires("thrust","tinyobjloader","tbb","libigl","eigen")
add_requires("cuda",{system = true })



target("main")




    add_files("src/**.cu")
    add_headerfiles("src/**.inl")
    add_includedirs("src/details")
    add_includedirs("src")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")
    add_cuflags("--expt-relaxed-constexpr")
    add_packages("cuda","cutlass","tinyobjloader","tbb","libigl","eigen")






