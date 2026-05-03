try:
    from src.env.material_pick_cube import MaterialPickCubeEnv, OBJECT_MATERIAL_PROFILES
except ModuleNotFoundError:
    MaterialPickCubeEnv = None
    OBJECT_MATERIAL_PROFILES = {}

__all__ = ["MaterialPickCubeEnv", "OBJECT_MATERIAL_PROFILES"]
