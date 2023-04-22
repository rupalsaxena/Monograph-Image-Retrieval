class ImgObj:
    def __init__(self, setting, scene, frame):
        self.setting = setting
        self.scene = scene
        self.frame = frame
        self.rgb = None
        self.depth = None
        self.semantic = None
    
    def set_rgb(self, rgb):
        self.rgb = rgb
    
    def set_depth(self, depth):
        self.depth = depth

    def set_sematic(self, semantic):
        self.semantic = semantic