

# Important Note

# this program is NOT a python standalone application
# it is inteded fror the GhPython component in Rhino 6 Grasshopper
# in this form it only serves the understanding of the algorithm


"""Provides a scripting component.
    Inputs:
        x: The x script variable
        y: The y script variable
    Output:
        a: The a output variable"""

__author__ = "Rolf"
__version__ = "2021.08.03"

import copy
import System
import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
import ghpythonlib.components as ghc
import clr
clr.AddReferenceToFileAndPath("C:\Program Files\Rhino 6\Plug-ins\Karamba\Karamba.gha")
clr.AddReferenceToFileAndPath("C:\Program Files\Rhino 6\Plug-ins\Karamba\KarambaCommon.dll")
import Karamba as ka
import Karamba.GHopper as kagh
import ghpythonlib.treehelpers as th
import KarambaCommon as kac
import ghpythonlib.treehelpers as th
import math
import time

xy_plane = rg.Plane(rg.Point3d(0,0,0),rg.Point3d(1,0,0),rg.Point3d(0,1,0))
xz_plane = rg.Plane(rg.Point3d(0,0,0),rg.Point3d(1,0,0),rg.Point3d(0,0,1))

# since ironpython doesn't have math.isclose() i need to set my own tolerance for comparisons 
tol = 1e-7
osteoblasts = []
osteoclasts = []
node_counter = 0
# faktor für die achsenverschiebung von 0 bis 1
shift_factor = 0.15
cluster_limit = 1

tmp_a = []
tmp_b = []
test_line = []
test_pt = []

class xmodel:
    '''object that stores and manages the properties of the geometry to optimize, including the karamba model'''

    def __init__(self, lines = None, frame = None, load_pts = None, supports = None):
        self.volume = None
        self.probes = []
        self.lines = lines
        #self.probes = None
        #self.line_division_planes = None
        #self.forces = None
        self.frame = frame
        self.load_pts = load_pts
        self.supports = supports
        self.ka_supports = None
        self.ka_loads = None
        self.ka_elements = None
        self.ka_model = None
        # forces is a list of lists where [0]normal [1]Mt [2]My [3]Mz

        if self.lines == None and self.frame == None and self.load_pts == None and self.supports == None:
            self.init_voronoi(frame_pt, voronoi_divisions, seed)
            # to ensure a certain minimal line length
    #        elements_length = [i.Length for i in self.lines]
    #        seed_tmp = seed
    #        while min(elements_length) < ((sum(elements_length)/len(elements_length))/10) and seed_tmp <= (seed+1000):
    #            seed_tmp += 1
    #            self.init_voronoi(frame_pt, voronoi_divisions, seed_tmp)
    #            elements_length = [i.Length for i in self.lines]
    #            print(seed_tmp)

        elif self.lines != None or self.frame != None or self.load_pts != None or self.supports != None:
            print('incomplete input for mode initialization')
        
        
    def init_voronoi(self, frame_pt, numer_of_divisions, seed):
        self.load_pts = []
        self.supports = []
        #3D
        if frame_pt.Y != 0:
            self.volume = True
            # creating the base frame and populating it
            self.frame = rg.Box(xy_plane, [rg.Point3d(0,0,0), frame_pt])
            voronoi_pts = ghc.Populate3D(self.frame, numer_of_divisions, seed)
            cells = ghc.Voronoi3D( voronoi_pts, self.frame)[0]
            # getting clean list for the nodes and lines
            tmp = ghc.DeconstructBrep(cells)
            self.lines = ghc.karambaGH.RemoveDuplicateLinesKaramba3D(tmp.edges)[0]
            #flipping the line by creating a new one with reversed start and endpoints
            #self.lines[2] = rg.Line(self.lines[2].PointAt(1),self.lines[2].PointAt(0))
            #global tmp_a
            #for line in self.lines:
            #    tmp_a.append(line.PointAt(0))
            #    tmp_a.append(line.PointAt(1))
            nodes_tmp = rg.Point3d.CullDuplicates(tmp.vertices, 0.001)
        #2D
        else:
            self.volume = False
            # creating the base frame and populating it
            self.frame = rg.Rectangle3d(xz_plane, rg.Point3d(0,0,0), frame_pt)
            voronoi_pts = ghc.Populate2D(self.frame, numer_of_divisions, seed)
            cells = ghc.Voronoi(voronoi_pts, boundary = self.frame)
            # getting clean list for the nodes and lines
            tmp = ghc.Explode(cells, True)
            self.lines = ghc.karambaGH.RemoveDuplicateLinesKaramba3D(tmp.segments)[0]
            nodes_tmp = rg.Point3d.CullDuplicates(tmp.vertices, 0.001)
        
        # load points are set to highest points and supports are set to lowest corner points
        
        x_list = [point.X for point in nodes_tmp]
        y_list = [point.Y for point in nodes_tmp]
        z_list = [point.Z for point in nodes_tmp]
        
        for point in nodes_tmp:
            if abs(point.Z - max(z_list)) < tol:
                self.load_pts.append(point)
            if abs(point.Z - min(z_list)) < tol:
                if abs(point.X - min(x_list)) < tol or abs(point.X - max(x_list)) < tol:
                    if abs(point.Y - min(y_list)) < tol or abs(point.Y - max(y_list)) < tol:
                        self.supports.append(point)
    
    def initialize_probes(self):
        # divides fillament into probe points
        points_tmp = []
        for line in self.lines:
            # convert lines to rhino.geometry
            line = rg.LineCurve(line)
            t_temp = line.DivideByCount(beam_divisions-1, True) 
            for t in t_temp:
                points_tmp.append(line.PointAt(t))
        
        for i in range(len(points_tmp)):
            self.probes.append(xprobe_point(points_tmp[i], i, initial_filament_radius))
        
        for j in range(len(self.probes)):
            self.probes[j].determine_node()
            #count_node returns True to increase node counter number for the next node 
            if self.probes[j].count_node() == True:
                global node_counter
                node_counter += 1
            
            self.probes[j].check_if_support()
            self.probes[j].check_if_load_pt()
        
    def update_probe_tensions(self):
        for i in range(len(self.probes)):
            self.probes[i].calc_tension()
        for j in range(len(self.probes)):
            if self.probes[j].node == True:
                self.probes[j].equalize_node() # necessary here?
                #self.probes[i].equalize_node_radii() # necessary here?
        
    def update_probe_positions(self):
        points_tmp = []
        for line in self.lines:
            # convert lines to rhino.geometry
            line = rg.LineCurve(line)
            t_temp = line.DivideByCount(beam_divisions-1, True) 
            for t in t_temp:
                points_tmp.append(line.PointAt(t))
        
        for i in range(len(points_tmp)):
            self.probes[i].point = points_tmp[i]
    
    def update_model_axis(self):
        global test_pt
        node_list = []
        new_lines = []
        
        for i in range(node_counter):
            tmp_pts_indices = []
            tmp_pts = None
            for j in range(len(self.probes)):
                if self.probes[j].node == True:
                    # maybe exclude loads and supports here
                    if self.probes[j].node_number == i:
                        tmp_pts_indices.append(j)
            for k in tmp_pts_indices:
                if self.probes[k].point_shiftet == None:
                    self.probes[k].point_shiftet = self.probes[k].point
                if tmp_pts == None:
                    tmp_pts = self.probes[k].point_shiftet
                tmp_pts += self.probes[k].point_shiftet
            new_node = tmp_pts/(len(tmp_pts_indices)+1)
            test_pt=new_node

            for m in tmp_pts_indices:
                self.probes[m].point = copy.deepcopy(new_node)
                
        for n in range(len(self.probes)):
                if self.probes[n].node == True:
                    node_list.append(n)
        for o in range(len(node_list)):
            if o % 2 == 0:
                new_lines.append(rg.Line(self.probes[node_list[o]].point, self.probes[node_list[o+1]].point))
        self.lines = new_lines
        self.update_probe_positions()
        self.update_probe_tensions()
                
    def reset_cluster_counter(self):
        for i in range(len(model.probes)):
            model.probes[i].targeted = 0
    # there are Karamba.Geometry and Karamba.GHopper.Geometry types for C# and GH envoirenment.  
    # most of the karamba geometry is done with ghpythonlib.components, which just mirrors the grasshopper-component, but still results in Karamba.Geometry.
    # after each function the Karamba.Geometry is converted to Karamba.GHopper.Geometry as required for the ghpythonlib model-assembly.
    # only the model and beam calculation is done within the Karamba.Geometry Namespace
    
    def get_ka_supports(self):
        # getting the points converted to karamba-supports and then to karambaGH-supports
        support_conditions = System.Collections.Generic.List[bool]()
        for i in range(6):
            support_conditions.Add(True)
        supports_tmp = []

        for i in range(node_counter):
            for j in range(len(self.probes)):
                if self.probes[j].support_pt == True and self.probes[j].node_number == i:
                    point_tmp = kagh.Geometry.GeometryExtensions.Convert(self.probes[j].point)
                    plane_tmp = copy.deepcopy(xy_plane)
                    plane_tmp = kagh.Geometry.GeometryExtensions.Convert(plane_tmp)
                    supports_tmp.append(ka.Supports.Support(point_tmp , support_conditions, plane_tmp ))
                    break
        self.ka_supports = [kagh.Supports.GH_Support(x) for x in supports_tmp]
        
    def get_ka_loads(self):
        loads_tmp = []
        #rotation
        #point_tmp = rg.Point3d(0,0,-100)
        #rotation = (100-iterations)*math.pi/180/8
        #point_tmp2 = ghc.Rotate(point_tmp, rotation, xz_plane)[0]
        # annoying conversion from gh to karamba
        f = kagh.Geometry.GeometryExtensions.Convert(rg.Vector3d(0,0,-100))  # originally rg.Vector3d(0,0,-100) for 100 kN
        m = kagh.Geometry.GeometryExtensions.Convert(rg.Vector3d(0,0,0))
        for i in range(node_counter):
            for j in range(len(self.probes)):
                if self.probes[j].load_pt == True and self.probes[j].node_number == i:
                    # more annoying conversion from gh to karamba
                    ka_point = copy.deepcopy(self.probes[j].point)
                    ka_point = kagh.Geometry.GeometryExtensions.Convert(ka_point)
                    loads_tmp.append(ka.Loads.PointLoad(ka_point, f, m, 0, False))
                    break
        # converting karamba-loads to karambaGH-loads
        self.ka_loads = [kagh.Loads.GH_Load(x) for x in loads_tmp]
        
    def get_ka_elements(self):
        # getting the lines converted to karamba-elements and then to karambaGH-elements
        elements_tmp = ghc.karambaGH.LinetoBeamKaramba3D(self.lines)[0]
        if len(elements_tmp) != len(self.lines):
            print('beam intersection detected') 
        self.ka_elements = [kagh.Elements.GH_Element(x) for x in elements_tmp]
        
    def calc_ka_model(self):
        
        self.get_ka_supports()
        self.get_ka_loads()
        self.get_ka_elements()
        model_tmp = ghc.karambaGH.AssembleModelKaramba3D(elem = self.ka_elements, support = self.ka_supports,  load = self.ka_loads)[0]
        self.ka_model = ka.Algorithms.ThIAnalyze.solve(model_tmp)[4]

        
        
    def calc_beam_forces(self):
        # returns forces in a list of lists where [0]normal force [1]Mt [2]My [3]Mz for every point
        tmp_list = []
        # creating a list for every point
        for i in range(len(self.probes)):
            tmp_list.append([])
        element_ids = System.Collections.Generic.List[str]()
        for element in self.ka_model.elems:
            element_ids.Add(element.id)
        #loop through elements of model
        tmp_lines_length = [line.Length for line in self.lines]
        model_forces = ka.Results.BeamForces.solve(self.ka_model, element_ids, -1, max(tmp_lines_length), beam_divisions)[0]
        # this whole part is a relic from an earlier version and could be done clearer
        for i in range(len(model_forces)):
            #loop through points on elements
            for j in range(len(model_forces[i])):
                normal_force = model_forces[i][j][0]
                tmp_list[(i*len(model_forces[i])+j)].append(abs(normal_force)) #inverting all negative normal forces(tensile stres) might distort results
                tmp_list[(i*len(model_forces[i])+j)].append(model_forces[i][j][3])
                tmp_list[(i*len(model_forces[i])+j)].append(model_forces[i][j][4])
                tmp_list[(i*len(model_forces[i])+j)].append(model_forces[i][j][5])
        for i in range(len(tmp_list)):
            self.probes[i].forces = tmp_list[i]


class xprobe_point:
    '''object that encapsulates several properties of a fillament probe point'''
    def __init__(self, point, index, radius):
        self.plane = None
        self.point = point
        self.locked = False
        self.support_pt = False
        self.load_pt = False
        self.point_shiftet = None
        self.index = index
        self.line_index = int(math.floor(self.index/beam_divisions)) # unsolved problem with indice 0
        self.forces = None
        self.radius = radius
        self.tension = None 
        self.tension_line = None
        self.node = False
        self.node_number = None
        self.targeted = 0
        
    def determine_node (self):
        # check if probe_point is a node
        for i in range(len(model.probes)):
            if self.index != model.probes[i].index and self.point == model.probes[i].point:
                self.node = True
    
    def check_if_support(self):
        
        if self.point in model.supports:
            self.support_pt = True
            self.locked = True
    
    def check_if_load_pt(self):
        
        if self.point in model.load_pts:
            self.load_pt = True
            self.locked = True
    
    def count_node (self):
        #checks for new unseen nodes, assigns current node_count number to all points of the node and returns True to increase node_counter
        for i in range(len(model.probes)):
            if model.probes[i].node_number == None and model.probes[i].node == True:
                node_group_indices = [j.index for j in model.probes if model.probes[i].point == j.point]
                for k in node_group_indices:
                    model.probes[k].node_number = node_counter
                return True
    
    def calc_tension(self):
        self.calc_tension_complete()
        self.calc_tension_lines()
    
    def calc_tension_normal_force(self):
        A = (math.pi * (self.radius ** 2))
        N = self.forces[0]
        self.tension = N/A  # tension is N/A at given probe point (A = pi*r²)
        
    def calc_tension_moment(self):
        M = math.sqrt(self.forces[2]**2+self.forces[3]**2)
        I = (math.pi*self.radius**4)/4
        self.tension = M/I
    
    def calc_tension_complete(self):
        A = (math.pi * (self.radius**2)) # A[cm²] = pi*r²[cm]
        #print(self.index)
        #print(self.point)
        #print(len(self.forces))
        N = self.forces[0] # [kN]
        M = math.sqrt(self.forces[2]**2+self.forces[3]**2)*100 # umrechnung von [kNm] in [kNcm]
        W = (math.pi*((self.radius*2)**3))/32
        self.tension = abs(N/A) + M/W # abs hier richtig?

    
    def get_local_ka_plane(self):
        element = model.ka_elements[int(math.floor(self.index/beam_divisions))]
        # I'm mapping the local coordinate system for each probe point according to a relation i found between karambas "disassemble elemement local coordinate system" and karambas beam forces Mx and My moments
        tmp_plane = ghc.karambaGH.DisassembleElementKaramba3D(element)[6]
        self.plane = rg.Plane(self.point, tmp_plane.ZAxis, rg.Vector3d.Negate(tmp_plane.YAxis))
    
    def calc_tension_lines(self):
        self.get_local_ka_plane()
        point = rg.Point3d(self.forces[2]/30, self.forces[3]/30, 0) # find better solution to scale tension lines/shift factor
        xform = rg.Transform.PlaneToPlane(xy_plane, self.plane)
        point.Transform(xform)
        self.tension_line = rg.Line(self.point ,point)
        
    # assign highest tension to all probe points at a node to prevent agents to get in a tug war over one node with diffrent inherit tesions
    def equalize_node(self):
        
        node_group_indices = [i.index for i in model.probes if self.node_number == i.node_number]
        node_group_tensions = [model.probes[j].tension for j in node_group_indices]
        node_group_radii = [model.probes[j].radius for j in node_group_indices]
        
        for k in node_group_indices:
            model.probes[k].radius = max(node_group_radii)
            model.probes[k].tension = max(node_group_tensions)
            
    # radius transformations to a single point of a node are transfered to the other points
    def equalize_current_node(self): # unused
        self.calc_tension()
        node_group_indices = [i.index for i in model.probes if self.node_number == i.node_number]
        #print(node_group_indices)
        for j in node_group_indices:
            model.probes[j].radius = self.radius
            model.probes[j].calc_tension()
    
    def update_after_shift(self): # old ?
        self.plane = new_plane
        self.point = self.plane.Origin
        self.calc_tension()
        self.equalize_node()



class xagent:
    '''base class for osteoblast and osteoclast''' 
    def __init__(self, point):
        self.point = point
        self.path_record = []
        
    def get_target(self):
        #check for the point with the highest/lowest tension within vision range
        tension_extreme = None
        target_index = None
        vision_tmp = agent_vision_radius
        
        while target_index == None:
            for i in range(len(model.probes)):
                if model.probes[i].targeted < cluster_limit:
                    if self.point.DistanceTo(model.probes[i].point) <= vision_tmp:
                        if self.osteoblast:
                            #osteoblast finding probe point with max tension
                            if tension_extreme == None or tension_extreme < model.probes[i].tension:
                                tension_extreme = model.probes[i].tension
                                target_index = i
                        if self.osteoclast:
                            #prevent osteoclast from freezing by targeting andtrying to decrease a probe point radius below minimum threshold 
                            if model.probes[i].radius > filament_radius_threshold*1.1:
                                #osteoclast finding probe point with min tension
                                if tension_extreme == None or tension_extreme > model.probes[i].tension:
                                    tension_extreme = model.probes[i].tension
                                    target_index = i
            vision_tmp += max_element_length/5
            if vision_tmp >= max_element_length*5 :
                print('infinite loop last bail')
                break
        
        if model.probes[target_index].node == True:
            node_group_indices = [i.index for i in model.probes if model.probes[target_index].node_number == i.node_number]
            for j in node_group_indices:
                model.probes[j].targeted += 1
        else:
            model.probes[target_index].targeted += 1
        return model.probes[target_index].point
        
    def move(self, targetpoint):
        # move agent in the direction of input target according to speed
        temp_move_vec = None
        temp_move_vec_scaled = None
        
        #prevent overshooting the targetpoint
        if self.point.DistanceTo(targetpoint) <= agent_speed:
            self.point = targetpoint
            self.path_record.append(targetpoint)
        #getting movement vector and magnitude from origin to targetpoint
        else:
            temp_move_vec = targetpoint - self.point
            temp_move_vec_scaled = rg.Vector3d.Multiply(temp_move_vec, (agent_speed/temp_move_vec.Length))
            self.point = self.point + temp_move_vec_scaled
            self.path_record.append(self.point)
            
    def get_closest_probe_index(self, originpoint = None):
        if originpoint == None:
            originpoint = self.point
        closest_point_index = None
        for i in range(len(model.probes)):
            if closest_point_index == None or originpoint.DistanceTo(model.probes[i].point) < originpoint.DistanceTo(model.probes[closest_point_index].point):
                closest_point_index = i
        return closest_point_index
        
    def work_closest(self):
        closest = self.get_closest_probe_index()

        # osteoblast increase cross-section radius at nearby probe points
        if self.point.DistanceTo(model.probes[closest].point) <= agent_effect_radius:
            agent_factor = 0
            if self.osteoblast == True:
                agent_factor = 1
            if self.osteoclast == True and model.probes[closest].radius >= filament_radius_threshold:
                agent_factor = -1
            if model.probes[closest].node == True:
                node_group_indices = [i.index for i in model.probes if model.probes[closest].node_number == i.node_number]

                for j in node_group_indices:
                    # the agent factor makes the osteoblast add and the osteoclast remove radius
                    model.probes[j].radius += agent_strength * agent_factor 
                    model.probes[j].calc_tension()
                model.probes[closest].equalize_node()
            else:
                # update the tension for new cross-section
                model.probes[closest].radius += agent_strength * agent_factor
                model.probes[closest].calc_tension()

    
    def axis_shift(self):
        global test_line 
        closest = self.get_closest_probe_index()
        line_index = model.probes[closest].line_index
        t = model.lines[line_index].ClosestParameter(model.probes[closest].point)
        if self.point.DistanceTo(model.probes[closest].point) <= agent_effect_radius:
            # check on which side of the fillament the agent is
            if t > 0.5:
                start = 0
                end = 1
            else:
                start = 1
                end = 0
            rotation_center = model.lines[line_index].PointAt(start)
            rotation_pt = model.lines[line_index].PointAt(end)
            # the probe point at the end of the current fillament is checked for support or load status, in which case it isn't moved
            rotation_probe_index = self.get_closest_probe_index(rotation_pt)
            if model.probes[rotation_probe_index].locked == False:
                rotation_vector1 = rotation_center - model.probes[closest].tension_line.PointAt(0)
                # the shift factor determines the degree of the rotation by moving the axis in the direction of the moment line 
                rotation_vector2 = rotation_center - model.probes[closest].tension_line.PointAt(shift_factor)
                # by rotation from the end to the start of the moment line we achieve a rotation in the opposite direction of the moment
                xform = rg.Transform.Rotation(rotation_vector2, rotation_vector1, rotation_center)
                # find the corresponding probe point at the end of the axis to rotate
                for i in range(len(model.probes)):
                    if line_index == model.probes[i].line_index and rotation_pt == model.probes[i].point:
                        if model.probes[i].point_shiftet == None:
                            model.probes[i].point_shiftet = copy.deepcopy(model.probes[i].point)
                        model.probes[i].point_shiftet.Transform(xform)

                        test_line= []
                        test_line.append(rg.Line(model.probes[closest].tension_line.PointAt(0),model.probes[closest].tension_line.PointAt(shift_factor)))
                        test_line.append(rg.Line(model.probes[i].point, model.probes[i].point_shiftet))

        
    def work_multiple (self):
        for i in range(len(model.probes)):
            if self.point.DistanceTo(model.probes[i].point) <= agent_effect_radius:
                # osteoblast increase cross-section radius at nearby probe points
                if self.osteoblast: 
                    model.probes[i].radius += agent_strength
                 # osteoclast decrease cross-section radius at nearby probe points
                elif model.probes[i].radius >= filament_radius_threshold:
                    model.probes[i].radius -= agent_strength
                #update the tension for new cross-section
                model.probes[i].calculate_tension()

class xosteoblast(xagent):
    '''Agent that moves to areas of highest tension and increases cross-section'''
    osteoblast = True
    osteoclast = False


class xosteoclast(xagent):
    '''Agent that moves to areas of highest tension and increases cross-section'''
    osteoblast = False
    osteoclast = True

# initialize model

model = xmodel()


elements_length = [i.Length for i in model.lines]
max_element_length = math.ceil(max(elements_length))




initial_filament_radius = max_element_length / 0.5
print(initial_filament_radius)
filament_radius_threshold = initial_filament_radius / 5  # danger: if the threshold is larger than the initial radius move() will produce an infinite loop
print(filament_radius_threshold)
agent_vision_radius = max_element_length * agent_vision_radius_factor / 2 # <3
print(agent_vision_radius)
agent_effect_radius = agent_vision_radius * agent_effect_radius_factor / 3.5
print(agent_effect_radius)
agent_speed = max_element_length / 20
print(agent_speed)

model.initialize_probes()
model.calc_ka_model()
model.calc_beam_forces()
model.update_probe_tensions()

#initialize agent startpoints


if model.volume == True:
    osteoclast_points = ghc.Populate3D(model.frame, osteoclast_number, seed)
    osteoblast_points = ghc.Populate3D(model.frame, osteoblast_number, seed+1)
else:
    osteoclast_points = ghc.Populate2D(model.frame, osteoclast_number, seed)
    osteoblast_points = ghc.Populate2D(model.frame, osteoblast_number, seed+1)

# initialize agent vision and effect radius




#initialize osteoblast instances

if type(osteoblast_points) is rg.Point3d:
    # ensure to get a list for a single agent to prevent type errors
    osteoblast_points = [osteoblast_points]  
for i in range(osteoblast_number):
    osteoblasts.append(xosteoblast(osteoblast_points[i]))

#initialize osteoclast instances

if type(osteoclast_points) is rg.Point3d:
    # ensure to get a list for a single agent to prevent type errors
    osteoclast_points = [osteoclast_points] 
for i in range(osteoclast_number):
    osteoclasts.append(xosteoclast(osteoclast_points[i]))

#put agents to work

for i in range(iterations):
    print('iteration' + str(i))
    for j in range(osteoblast_number):
        osteoblasts[j].move(osteoblasts[j].get_target())
        osteoblasts[j].work_closest()
        osteoblasts[j].axis_shift()
    for k in range(osteoclast_number):
        osteoclasts[k].move(osteoclasts[k].get_target())
        osteoclasts[k].work_closest()
        osteoclasts[k].axis_shift()
    model.reset_cluster_counter()
    model.update_model_axis()
    model.calc_ka_model()
    model.calc_beam_forces()
    #for i in range(len(model.probes)):
    #    model.probes[i].forces = model.forces[i]
    #    model.probes[i].update_after_shift(model.line_division_planes[i])
    

radii_out = []
probe_points_out = []
tensions_out = []
tension_lines_out = []
planes_out = []
moment_res_out = []

for i in range(len(model.probes)):
    radii_out.append(model.probes[i].radius)
    probe_points_out.append(model.probes[i].point)
    tensions_out.append(model.probes[i].tension)
    tension_lines_out.append(model.probes[i].tension_line)
    planes_out.append(model.probes[i].plane)
    moment_res_out.append(math.sqrt(model.probes[i].forces[2]**2+model.probes[i].forces[3]**2))

probe_points_out = th.list_to_tree(probe_points_out)

#export osteoblasts
osteoblast_targets_out = []
osteoblast_paths_out = []
for i in range(osteoblast_number):
    osteoblast_paths_out.append(osteoblasts[i].path_record)
    osteoblast_targets_out.append(osteoblasts[i].get_target())
osteoblast_paths_out = th.list_to_tree(osteoblast_paths_out)


#export osteoclasts
osteoclast_targets_out = []
osteoclast_paths_out = []
for i in range(osteoclast_number):
    osteoclast_paths_out.append(osteoclasts[i].path_record)
    osteoclast_targets_out.append(osteoclasts[i].get_target())
osteoclast_paths_out = th.list_to_tree(osteoclast_paths_out)
osteoclast_targets_out =th.list_to_tree(osteoclast_targets_out)


ka_model_out = ka.GHopper.Models.GH_Model(model.ka_model)
ka_supports_out = model.ka_supports
ka_loads_out = model.ka_loads
ka_elements_out = model.ka_elements
load_pts_out = model.load_pts

lines_out = model.lines

a1 = model.lines
a2 = model.lines
a3 = model.supports



#        self.plane = None
#        self.point = point
#        self.locked = False
#        self.support_pt = False
#        self.load_pt = False
#        self.point_shiftet = None
#        self.index = index
#        self.line_index = int(math.floor(self.index/beam_divisions)) # unsolved problem with indice 0
#        self.forces = None
#        self.radius = radius
#        self.tension = None 
#        self.tension_line = None
#        self.node = False
#        self.node_number = None