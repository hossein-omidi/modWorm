from juliacall import Main as jl
print(jl.seval("VERSION"))
print(jl.seval("Base.active_project()"))
jl.seval("import Pkg; Pkg.status()")


jl.seval('import Pkg; Pkg.add("DifferentialEquations")')
jl.seval('import Pkg; Pkg.add("Sundials")')
jl.seval('import Pkg; Pkg.add("OrdinaryDiffEq")')

jl.seval("using DifferentialEquations, Sundials, OrdinaryDiffEq")
print("Julia packages loaded")