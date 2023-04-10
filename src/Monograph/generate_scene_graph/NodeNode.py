# store Node to Node data in form of object
class NodeNode:
    def __init__(self, fr_cord, to_cord, fr_id, to_id, frame_id=None, scene_id=None):
        self.from_coord = fr_cord
        self.to_coord = to_cord
        self.from_id = fr_id
        self.to_id = to_id
        self.frame_id = frame_id
        self.scene_id = scene_id