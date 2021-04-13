from dolfin import *
import mshr
import numpy as np
import os
from os import path

import sys
if len(sys.argv)==1:
   info("Input parameter is missing. Run the script using 'python3 kelvin-voigt_temperature.py input_parameters'.")
   sys.exit()

parameters_file = sys.argv[1]

if not path.exists(parameters_file):
   info("File '"+parameters_file+"' does not exist.")
   sys.exit()

comm = MPI.comm_world
rank = MPI.rank(comm)
set_log_level(LogLevel.INFO if rank==0 else LogLevel.INFO)
parameters["std_out_all_processes"] = False
parameters["form_compiler"]["quadrature_degree"] = 8

geometryhalf = mshr.Rectangle(Point(0.0, 0.0065), Point(0.080, 0.013)) \
           - mshr.Rectangle(Point(0.072, 0.0064), Point(0.080, 0.0066)) \
	   - mshr.Circle(Point(0.072, 0.0065), 0.0001, 60)

# Build mesh
mesh = mshr.generate_mesh(geometryhalf, 100)
mesh.init()
    
# Construct facet markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
bndry.set_all(0)
    
for f in facets(mesh):
        if f.exterior():
            mp = f.midpoint()
            #if near(mp[0], -1.0) and near(mp[1], 1.0): bndry[f] = 1  # left upper corner
            if near(mp[0], 0.0): bndry[f] = 1  # left
            elif near(mp[0], 0.080): bndry[f] = 2  # right
            elif near(mp[1], 0.0): bndry[f] = 3  # bottom
            elif near(mp[1], 0.013): bndry[f] = 4  # top
            elif near(mp[1], 0.0065): bndry[f] = 5  # halfbottom

meshes=[(mesh,bndry)]
    
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
for k in range(1):
        info("refinement level {}".format(k))
        #cf = CellFunction('bool', mesh, False)
        cf = MeshFunction("bool", mesh, mesh.topology().dim(), 0.0)
        for c in cells(mesh):
            #if c.midpoint().distance(center)<0.06 : cf[c]=True
            cf[c]=True
	
        mesh=refine(mesh,cf,redistribute=False)
        bndry=adapt(bndry,mesh)
        for f in facets(mesh):
            if bndry[f]>10 : bndry[f]=0
        meshes.append((mesh,bndry))

#local refinement
for k in range(2):
        info("refinement level {}".format(k))
        #cf = CellFunction('bool', mesh, False)
        cf = MeshFunction("bool", mesh, mesh.topology().dim(), 0.0)
        for c in cells(mesh):
            for vert in vertices(c):
                if (vert.point()[0]>0.07149 and vert.point()[1]<0.006751 and vert.point()[1]>0.006399) : cf[c]=True
                elif (vert.point()[0]>0.07 and vert.point()[0]<0.075 and vert.point()[1]<0.0075) : cf[c]=True
	
        mesh=refine(mesh,cf,redistribute=False)
        bndry=adapt(bndry,mesh)
        for f in facets(mesh):
            if bndry[f]>10 : bndry[f]=0
        meshes.append((mesh,bndry))
        
# Define finite elements
Eu = VectorElement("CG", mesh.ufl_cell(), 1)
Ev = VectorElement("CG", mesh.ufl_cell(), 1)
Eth = FiniteElement("CG", mesh.ufl_cell(), 1)

# Build function spaces (Taylor-Hood)
W = FunctionSpace(mesh, MixedElement([Eu, Ev, Eth]))

# No-slip boundary condition for velocity on walls and cylinder - boundary id 3
zero = Constant(0)
zerovec = (0, 0)
onevecu =  Expression(("0.01*time","0"), time = 0.0, degree = 1)
onevecv = (0.01, 0)

prestop1 =  Expression("0.0033*time", time = 0.0, degree = 1)
prestop2 =  Constant(0.0033)

bc_bottomhalf1 = DirichletBC(W.sub(0).sub(1), zero, bndry, 5) #bottom half u
bc_bottomhalf2 = DirichletBC(W.sub(1).sub(1), zero, bndry, 5) #bottom half v

bc_top1 = DirichletBC(W.sub(0).sub(1), prestop1, bndry, 4) #top u
bc_top2 = DirichletBC(W.sub(1).sub(1), prestop2, bndry, 4) #top v

bcs = [bc_bottomhalf1, bc_bottomhalf2, bc_top1, bc_top2]

u_, v_, th_ = TestFunctions(W)
w = Function(W)
u, v, th = split(w)

#previous time step
w0 = Function(W)
(u0, v0, th0) = split(w0)

exec(open(parameters_file).read())

I = Identity(mesh.geometry().dim())
F = I + grad(u)
B = F*F.T
J = det(F)
Bbar = B/J
FmT = inv(F.T)
Fm = inv(F)
Dtilde = (grad(v)*Fm + FmT*(grad(v).T))/2.0
Ttilde = 2.0*nu*Dtilde + lambd*tr(Dtilde)*I + mu1*(Bbar-0.5*tr(Bbar)*I)/J+K/(beta*J)*(1.0-J**(-beta))*I

Eq1 = inner(v - (u-u0)/dt,v_)*dx
Eq2 = (rhoR*inner((v-v0)/dt, u_) + inner(J*Ttilde*FmT, grad(u_)))*dx
Eq3 = ((rhoR*cV-th/(beta*beta)*d2K*(beta*ln(J)+J**(-beta)-1.0)-0.5*th*d2mu1*(tr(Bbar)-2.0))*(th - th0)*th_/dt + inner(kappa*J*Fm*FmT*grad(th), grad(th_)) - J*((2.0*nu*inner(Dtilde, Dtilde) + lambd*tr(Dtilde)*tr(Dtilde))*th_)\
      - (th*dmu1*(inner(Bbar, Dtilde)-0.5*tr(Bbar)*tr(Dtilde)) - th*dK*tr(Dtilde)*(1.0-J**(-beta))/beta)*th_)*dx

Eq = Eq1 + Eq2 + Eq3

info("Solving problem of size: {0:d}".format(W.dim()))
problem=NonlinearVariationalProblem(Eq,w,bcs,derivative(Eq,w))
solver=NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 5e-9
solver.parameters['newton_solver']['relative_tolerance'] = 5e-9
solver.parameters['newton_solver']['maximum_iterations'] = 20
#solver.solve()

# Save solution in VTK format
ufile = XDMFFile(output_directory + '/u.xdmf')
vfile = XDMFFile(output_directory + '/v.xdmf')
thfile = XDMFFile(output_directory + '/th.xdmf')

info("dim= {}".format(W.dim()))

w0ic = Expression(("0.0","0.0","0.0","0.0","300.0"), degree = 1)
w0.assign(interpolate(w0ic, W))
w.assign(interpolate(w0ic, W))
(u0, v0, th0) = w0.split()

(u, v, th) = w.split()
u.rename("u", "displacement")
v.rename("v", "velocity")
th.rename("th", "temperature")
# Save to file
ufile.write(u, 0)
vfile.write(v, 0)
thfile.write(th, 0)

ufile.parameters["flush_output"] = True
vfile.parameters["flush_output"] = True
thfile.parameters["flush_output"] = True

#tic()
# Time-stepping
t = float(dt)

# Save temperature data at given measurement points to a file
temp_data = open(output_directory + '/temperature.csv', 'w')
temp_data.write('# Temperature data at given measurement points\n')
temp_data.write('"Time [s]","Temperature A [K]", "Temperature B [K]", "Temperature C [K]"\n') # Column headers

def create_list(leftbottom, righttop, Nx, Ny):
    p00 = leftbottom[0]
    p01 = leftbottom[1]
    p10 = righttop[0]
    p11 = righttop[1]
    deltax = (p10 - p00)/Nx
    deltay = (p11 - p01)/Ny
    
    mesh.init()
    mesh.bounding_box_tree().build(mesh)
    listcoord=[]
    xcoord = p00
    while xcoord < p10 + 1e-6:
       ycoord = p01
       while ycoord < p11 + 1e-6:
          if mesh.bounding_box_tree().compute_first_entity_collision(Point(xcoord, ycoord)) <= mesh.num_cells():
             listcoord.append([xcoord, ycoord])
          ycoord = round(ycoord + deltay, 9)
       xcoord = round(xcoord + deltax, 9)
    
    return listcoord

def output_fields(listcoord, time):
    fields_data = open(output_directory + '/fields_time' + str(time) + '.csv', 'w')
    i = 0
    prevx = listcoord[0][0]
    while i < len(listcoord):
       #if listcoord[i][0] != prevx: fields_data.write(f'\n') #prida volny radek
       prevx = listcoord[i][0]
       fields_data.write(f'{str(listcoord[i])[1:-1]}, {th(Point(listcoord[i]))}, {u(Point(listcoord[i]))[0]}, {u(Point(listcoord[i]))[1]}\n')
       i += 1
    fields_data.close()
    
    return 0


listcoord = create_list(leftbottom, righttop, Nx, Ny)

while t < t_end:
    if t < 2.00001:
      prestop1.time = min(2.0, t)
      prestop2.assign(0.0033)
    elif t < 4.00001:
      prestop1.time = 2.0 - (t - 2.0)
      prestop2.assign(-0.0033)
    #elif t < 5.00001:
      #prestop1.time = 0.0
      #prestop2.assign(0.0)
      ##prestop1.time = t
      ##prestop2.time = t
    #elif t < 7.00001:
      #prestop1.time = min(2.0, t - 5.0)
      #prestop2.assign(0.0033)
      ##prestop1.time = t
      ##prestop2.time = t
    #elif t < 9.00001:
      #prestop1.time = 2.0 - (t - 7.0)
      #prestop2.assign(-0.0033)
      ##prestop1.time = t
      ##prestop2.time = t
    #elif t < 10.00001:
      #prestop1.time = 0.0
      #prestop2.assign(0.0)
      ##prestop1.time = t
      ##prestop2.time = t
    #elif t < 12.00001:
      #prestop1.time = min(2.0, t - 10.0)
      #prestop2.assign(0.0033)
      ##prestop1.time = t
      ##prestop2.time = t
    #elif t < 14.00001:
      #prestop1.time = 2.0 - (t - 12.0)
      #prestop2.assign(-0.0033)
      ##prestop1.time = t
      ##prestop2.time = t
    #elif t < 15.00001:
      #prestop1.time = 0.0
      #prestop2.assign(0.0)
      ##prestop1.time = t
      ##prestop2.time = t
    else:
      prestop1.time = 0.0
      prestop2.assign(0.0)
    
    info("t = {}".format(t))

    # Compute
    its, ok = solver.solve()
    
    # Extract solutions:
    (u, v, th) = w.split()
    u.rename("u", "displacement")
    v.rename("v", "velocity")
    th.rename("th", "temperature")
    # Save to file
    ufile.write(u, t)
    vfile.write(v, t)
    thfile.write(th, t)

    temp1 = th(0.071899, 0.0065) # Temperature measurement site C, crack tip
    temp2 = th(0.001, 0.011) # Temperature measurement site A
    temp3 = th(0.079, 0.011) # Temperature measurement site B

    temp_data.write(f'{t}, {temp2}, {temp3}, {temp1}\n') # Write temperature data to the file
    if abs(t-1.0)<1e-6: output_fields(listcoord, t)
    if abs(t-2.0)<1e-6: output_fields(listcoord, t)
    if abs(t-2.5)<1e-6: output_fields(listcoord, t)
    if abs(t-10.0)<1e-6: output_fields(listcoord, t)
    
    # Move to next time step[
    w0.assign(w)

    if t > 4.399:
      dt.assign(0.04)

    if t > 6.399:
      dt.assign(0.1)

    if t > 19.999:
      dt.assign(0.5)
      
    if t > 39.999:
      dt.assign(2.0)

    if t > 99.999:
      dt.assign(10.0)

    if t > 399.999:
      dt.assign(50.0)

    if t > 999.999:
      dt.assign(200.0)

    t = round(float(t + dt), 4)
    if t > t_end:
      dt.assign(t_end - t +float(dt))
      t = t_end

temp_data.close() # close file with temperature data at measurement points

