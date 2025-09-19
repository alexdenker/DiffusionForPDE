import warnings
warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import os 
import math
import sys

if not os.path.exists("data/mesh"):
    os.makedirs("data/mesh")

mesh_name = "disk" # "square"  "disk" "L"
mesh_type = "coarse" # "dense" "coarse"
show_mesh = False


if mesh_name == "disk":
   ### Parameters of mesh
   radius = 1.0
   if mesh_type == "dense":
      num_points = 128
   else:
      num_points = 92

   ### Start building mesh
   gmsh.initialize()
   gmsh.model.add("Disk")

   tag_list = []

   phi = np.linspace(0, 2*math.pi, num_points, endpoint=False)
   for i in range(num_points):
      #print("point: ", radius*np.cos(phi[i]), radius*np.sin(phi[i]))
      tag_list.append(gmsh.model.occ.addPoint(radius*np.cos(phi[i]), 
                                                radius*np.sin(phi[i]), 0.0, 
                                                meshSize=2*math.pi/num_points))

   for i in range(num_points-1):
      gmsh.model.occ.addLine(tag_list[i], tag_list[i+1])

   gmsh.model.occ.addLine(tag_list[-1], tag_list[0])

   loop = gmsh.model.occ.addCurveLoop(tag_list)
   surf = gmsh.model.occ.addPlaneSurface([loop])

   gmsh.model.occ.synchronize()

   ### Mark subdomains
   gmsh.model.addPhysicalGroup(2, [surf], 1, name="domain")

   gmsh.model.mesh.generate(2)
   #gmsh.write('disk.msh')
   #gmsh.fltk.initialize()

   gmsh.write(f"data/mesh/disk_{mesh_type}.msh")
   #gmsh.write("disk.png")  # Save as PNG
   #gmsh.write("disk.pdf")  # Save as PDF
   # Launch the GUI to see the results:
   #if show_mesh and '-nopopup' not in sys.argv:
   #   gmsh.fltk.run()

   # close gmsh
   gmsh.finalize()

elif mesh_name == 'square':

   ### Start building mesh
   gmsh.initialize()


   # cube points:
   if mesh_type == "dense":
      num_elements = 64
   else:
      num_elements = 32
   num_points = num_elements + 1
   lc = 1./num_points 

   # Corners
   point1 = gmsh.model.geo.add_point(0, 0, 0, lc)
   point2 = gmsh.model.geo.add_point(1, 0, 0, lc)
   point3 = gmsh.model.geo.add_point(1, 1, 0, lc)
   point4 = gmsh.model.geo.add_point(0, 1, 0, lc)


   lists = []
   lists.append(gmsh.model.geo.addLine(point1, point2))
   lists.append(gmsh.model.geo.addLine(point2, point3))
   lists.append(gmsh.model.geo.addLine(point3, point4))
   lists.append(gmsh.model.geo.addLine(point4, point1))

   loop = gmsh.model.geo.addCurveLoop(lists)
   surf = gmsh.model.geo.addPlaneSurface([loop])


   gmsh.model.geo.mesh.setTransfiniteSurface(surf, "Left", [point1, point2, point3, point4])

   gmsh.model.geo.synchronize()

   gmsh.model.addPhysicalGroup(2, [surf], 1, name="domain")

   gmsh.model.mesh.generate(2)

   # Write mesh data:
   #gmsh.write("square.msh")
   gmsh.fltk.initialize()
   gmsh.write(f"data/mesh/square_{mesh_type}.msh")
   # Launch the GUI to see the results:
   if show_mesh and '-nopopup' not in sys.argv:
      gmsh.fltk.run()

   # close gmsh
   gmsh.finalize()

elif mesh_name == "L":
   ### Start building mesh
   gmsh.initialize()


   # cube points:
   if mesh_type == "dense":
      num_elements = 64
   else:
      num_elements = 32
   num_points = num_elements + 1
   lc = 1./num_points 

   # Corners
   point1 = gmsh.model.geo.add_point(0, 0, 0, lc)
   point2 = gmsh.model.geo.add_point(1, 0, 0, lc)
   point3 = gmsh.model.geo.add_point(1.0, 0.5, 0, lc)
   point4 = gmsh.model.geo.add_point(0.5, 0.5, 0, lc)
   point5 = gmsh.model.geo.add_point(0.5, 1, 0, lc)
   point6 = gmsh.model.geo.add_point(0, 1, 0, lc)
   #point7 = gmsh.model.geo.add_point(0, 0.5, 0, lc)


   lists = []
   lists.append(gmsh.model.geo.addLine(point1, point2))
   lists.append(gmsh.model.geo.addLine(point2, point3))
   lists.append(gmsh.model.geo.addLine(point3, point4))
   lists.append(gmsh.model.geo.addLine(point4, point5))
   lists.append(gmsh.model.geo.addLine(point5, point6))
   lists.append(gmsh.model.geo.addLine(point6, point1))

   loop = gmsh.model.geo.addCurveLoop(lists)
   surf = gmsh.model.geo.addPlaneSurface([loop])

   #gmsh.model.geo.mesh.setTransfiniteSurface(surf, "Left", [point1, point2, point3, point4])

   gmsh.model.geo.synchronize()

   gmsh.model.addPhysicalGroup(2, [surf], 1, name="domain")

   gmsh.model.mesh.generate(2)

   # Write mesh data:
   #gmsh.write("L.msh")
   gmsh.fltk.initialize()
   #gmsh.write("L.png")  # Save as PNG
   #gmsh.write("L.pdf")  # Save as PDF
   gmsh.write(f"data/mesh/L_{mesh_type}.msh")

   # Launch the GUI to see the results:
   if show_mesh and '-nopopup' not in sys.argv:
      gmsh.fltk.run()

   # close gmsh
   gmsh.finalize()