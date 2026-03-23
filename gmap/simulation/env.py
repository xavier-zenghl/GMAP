"""SAPIEN simulation environment wrapper. Requires sapien >= 2.0."""

try:
    import sapien.core as sapien
    HAS_SAPIEN = True
except ImportError:
    HAS_SAPIEN = False

class ArticulatedEnv:
    def __init__(self, timestep=1/240):
        if not HAS_SAPIEN:
            raise ImportError("SAPIEN is required for simulation. Install with: pip install sapien")
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(timestep)
        self.scene.add_ground(altitude=0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.articulation = None
        self.robot = None

    def load_articulated_object(self, urdf_path):
        loader = self.scene.create_urdf_loader()
        self.articulation = loader.load(urdf_path)

    def load_robot(self, urdf_path, initial_qpos):
        loader = self.scene.create_urdf_loader()
        self.robot = loader.load(urdf_path)
        self.robot.set_qpos(initial_qpos)

    def get_joint_state(self):
        if self.articulation:
            return self.articulation.get_qpos()[0]
        return 0.0

    def step(self):
        self.scene.step()

    def close(self):
        self.scene = None
