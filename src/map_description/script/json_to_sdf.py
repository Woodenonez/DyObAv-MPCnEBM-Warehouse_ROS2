import json
import math
import xml.etree.ElementTree as ET

from typing import TypedDict


"""Generate SDF model and the config file for the map from a json file"""


class ObstacleInfoCompulsive(TypedDict):
    id_: int
    vertices: list[tuple[float, float]]

class ObstacleInfo(ObstacleInfoCompulsive, total=False):
    name: str

def calculate_wall_parameters(start, end, wall_height, wall_thickness):
    length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    yaw = math.atan2(end[1] - start[1], end[0] - start[0])
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    mid_z = wall_height / 2
    return (length, mid_x, mid_y, mid_z, yaw)

def create_wall_link(id, length, x, y, z, yaw, wall_height, wall_thickness):
    link = ET.Element('link', name=f'wall_{id}')
    ET.SubElement(link, 'pose').text = f'{x} {y} {z} 0 0 {yaw}'
    ET.SubElement(link, 'self_collide').text = '0'
    ET.SubElement(link, 'kinematic').text = '0'
    visual = ET.SubElement(link, 'visual', name=f'wall_{id}_visual')
    collision = ET.SubElement(link, 'collision', name=f'wall_{id}_collision')

    ### Visual ###
    geometry = ET.SubElement(visual, 'geometry')
    box = ET.SubElement(geometry, 'box')
    ET.SubElement(box, 'size').text = f'{length} {wall_thickness} {wall_height}'
    
    material = ET.SubElement(visual, 'material')
    script = ET.SubElement(material, 'script')
    ET.SubElement(script, 'uri').text = 'file://media/materials/scripts/gazebo.material'
    ET.SubElement(script, 'name').text = 'Gazebo/Grey'
    ET.SubElement(material, 'ambient').text = '1 0.764706 0.305882 1' # color in shadow
    ET.SubElement(material, 'diffuse').text = '0.7 0.7 0.7 1' # color in light
    ET.SubElement(material, 'specular').text = '0.01 0.01 0.01 1' # color in highlight
    ET.SubElement(material, 'emissive').text = '0 0 0 1' # color of light emitted
    ET.SubElement(material, 'shader', type='pixel')

    ET.SubElement(visual, 'cast_shadows').text = '1'
    ET.SubElement(visual, 'transparency').text = '0'

    ### Collision ###
    geometry = ET.SubElement(collision, 'geometry')
    box = ET.SubElement(geometry, 'box')
    ET.SubElement(box, 'size').text = f'{length} {wall_thickness} {wall_height}'

    ET.SubElement(collision, 'laser_retro').text = '0'
    ET.SubElement(collision, 'max_contacts').text = '10'

    surface = ET.SubElement(collision, 'surface')
    friction = ET.SubElement(surface, 'friction')
    ode = ET.SubElement(friction, 'ode')
    ET.SubElement(ode, 'mu').text = '1'
    ET.SubElement(ode, 'mu2').text = '1'
    ET.SubElement(ode, 'fdir1').text = '0 0 0'
    ET.SubElement(ode, 'slip1').text = '0'
    ET.SubElement(ode, 'slip2').text = '0'
    bounce = ET.SubElement(surface, 'bounce')
    ET.SubElement(bounce, 'restitution_coefficient').text = '0'
    ET.SubElement(bounce, 'threshold').text = '1000000' # 1e6
    contact = ET.SubElement(surface, 'contact')
    ET.SubElement(contact, 'collide_without_contact').text = '0'
    ET.SubElement(contact, 'collide_without_contact_bitmask').text = '1'
    ET.SubElement(contact, 'collide_bitmask').text = '1'
    ode = ET.SubElement(contact, 'ode')
    ET.SubElement(ode, 'soft_cfm').text = '0'
    ET.SubElement(ode, 'soft_erp').text = '0.2'
    ET.SubElement(ode, 'kp').text = '10000000000000.0' # 1e13
    ET.SubElement(ode, 'kd').text = '1'
    ET.SubElement(ode, 'max_vel').text = '0.01'
    ET.SubElement(ode, 'min_depth').text = '0'
    bullet = ET.SubElement(contact, 'bullet')
    ET.SubElement(bullet, 'split_impulse').text = '1'
    ET.SubElement(bullet, 'split_impulse_penetration_threshold').text = '-0.01'
    ET.SubElement(bullet, 'soft_cfm').text = '0'
    ET.SubElement(bullet, 'soft_erp').text = '0.2'
    ET.SubElement(bullet, 'kp').text = '10000000000000.0' # 1e13
    ET.SubElement(bullet, 'kd').text = '1'

    return link

def parse_json(json_file_path) -> list[list]:
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    boundary_coords = data['boundary_coords']
    if 'obstacle_dict' in data:
        obstacle_dict_list = data['obstacle_dict']
    else:
        obstacle_coords_list = data['obstacle_list']
        obstacle_dict_list = [ObstacleInfo(id_=i, name=f'obstacle_{i}', vertices=obs) for i, obs in enumerate(obstacle_coords_list)]

    wall_list = []
    for obstacle_dict in obstacle_dict_list:
        for i in range(len(obstacle_dict['vertices'])):
            start = obstacle_dict['vertices'][i]
            end = obstacle_dict['vertices'][(i+1)%len(obstacle_dict['vertices'])]
            wall_list.append([start, end])
    for i in range(len(boundary_coords)):
        start = boundary_coords[i]
        end = boundary_coords[(i+1)%len(boundary_coords)]
        wall_list.append([start, end])
    return wall_list

def create_sdf_model(json_file, sdf_file, wall_height=2.0, wall_thickness=0.1, model_name='default'):
    wall_list = parse_json(json_file)

    model = ET.Element('sdf', version='1.6')
    model_element = ET.SubElement(model, 'model', name=model_name)

    ET.SubElement(model_element, 'static').text = 'true'

    for idx, wall in enumerate(wall_list):
        length, x, y, z, yaw = calculate_wall_parameters(*wall, wall_height, wall_thickness)
        link = create_wall_link(idx, length, x, y, z, yaw, wall_height, wall_thickness)
        model_element.append(link)

    tree = ET.ElementTree(model)
    ET.indent(tree, space="\t", level=0)
    tree.write(sdf_file, encoding='utf-8', xml_declaration=True)

def create_config_file(config_file_path: str, model_name: str, author_name='Ze', author_email='zhze.zhang@gmail.com', description='Placeholder'):
    model = ET.Element('model')
    ET.SubElement(model, 'name').text = model_name
    ET.SubElement(model, 'version').text = '1.0'
    ET.SubElement(model, 'sdf', version='1.6').text = 'model.sdf'
    author = ET.SubElement(model, 'author')
    ET.SubElement(author, 'name').text = author_name
    ET.SubElement(author, 'email').text = author_email
    ET.SubElement(model, 'description').text = description

    tree = ET.ElementTree(model)
    ET.indent(tree, space="\t", level=0)
    tree.write(config_file_path, encoding='utf-8', xml_declaration=True)


# Example Usage
if __name__ == '__main__':
    import os
    import pathlib

    map_name = 'test_map' # the map name is the same as the json file name

    json_file_name = map_name + '.json'
    sdf_file_name = 'model.sdf'
    config_file_name = 'model.config'

    pkg_dir = pathlib.Path(__file__).resolve().parents[1]
    json_path = os.path.join(pkg_dir, 'data', json_file_name)
    sdf_path = os.path.join(pkg_dir, 'model', map_name, sdf_file_name)
    config_path = os.path.join(pkg_dir, 'model', map_name, config_file_name)
    
    create_sdf_model(json_path, sdf_path, model_name=map_name)
    create_config_file(config_path, model_name=map_name)

    print(f'Created map model file {sdf_path} from json file {json_path}')
