import bpy
import sys
from time import time
import random
from mathutils import Color

# Helpful resource:
# https://thousandyardstare.de/blog/generating-depth-images-in-blender-279.html

if __name__ == "__main__":
    obj_file = sys.argv[-3]
    output_prefix = sys.argv[-2]
    resolution = int(sys.argv[-1])

    random.seed(int(time()))

    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution

    bpy.ops.import_scene.obj(filepath=obj_file)
    cloth = bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action='DESELECT')
    cloth.select_set(True)
    plane_material = bpy.data.materials.get("Plane")
    node = plane_material.node_tree.nodes.get("Musgrave Texture")
    node.inputs[1].default_value = random.uniform(-100, 100)
    cloth_material = bpy.data.materials.get("Cloth")
    cloth_bsdf = cloth_material.node_tree.nodes.get("Principled BSDF")
    c = Color()
    c.hsv = (random.uniform(0.0, 1.0),
             random.uniform(0.0, 1.0),
             random.uniform(0.5, 1.0))
    cloth_bsdf.inputs[0].default_value = [
        c.r,
        c.g,
        c.b,
        1.0
    ]
    cloth.data.materials[0] = cloth_material

    # Shading smooth
    for f in cloth.data.polygons:
        f.use_smooth = True

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    composite_node = tree.nodes["Composite"]
    render_layers_node = tree.nodes["Render Layers"]

    # Save color
    output_node = tree.nodes["File Output"]
    output_node.base_path = output_prefix
    bpy.ops.render.render(write_still=True)
    exit()
