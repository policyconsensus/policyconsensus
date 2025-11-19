import numpy as np
import torch
from typing import Optional, List, Dict, Any
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import ObjectType
import time

from modular_policy.env.rlbench.env import RlbenchEnv


class HighFrictionRlbenchEnv(RlbenchEnv):
    """
    RLBench environment wrapper that increases friction coefficients for objects and gripper.
    This can help with more realistic grasping and manipulation behavior.
    """
    
    def __init__(self, 
                 object_friction: float = 2.0,
                 gripper_friction: float = 3.0,
                 table_friction: float = 1.5,
                 verbose: bool = True,
                 **kwargs):
        """
        Args:
            object_friction: Friction coefficient for manipulatable objects
            gripper_friction: Friction coefficient for gripper components  
            table_friction: Friction coefficient for table/surface
            verbose: Print friction modification details
        """
        super().__init__(**kwargs)
        
        self.object_friction = object_friction
        self.gripper_friction = gripper_friction
        self.table_friction = table_friction
        self.verbose = verbose
        
        # Keep track of modified objects
        self.modified_objects = []
        
    def reset(self, seed=None, options=None):
        """Reset environment and apply friction modifications"""
        obs, info = super().reset(seed=seed, options=options)
        
        # Apply friction modifications after environment reset
        self._modify_friction()
        
        return obs, info
    
    def _modify_friction(self):
        """Modify friction coefficients for objects and gripper"""
        if self.verbose:
            print("Applying friction modifications...")
            
        self.modified_objects = []
        
        # Get all shapes in the scene
        try:
            all_shapes = Shape.get_all_objects()
            
            for shape in all_shapes:
                shape_name = shape.get_name().lower()
                original_friction = self._get_current_friction(shape)
                
                if original_friction is None:
                    continue
                    
                # Modify gripper components
                if self._is_gripper_component(shape_name):
                    self._set_friction(shape, self.gripper_friction, "gripper")
                
                # Modify table/surface
                elif self._is_table_surface(shape_name):
                    self._set_friction(shape, self.table_friction, "table")
                
                # Modify manipulatable objects
                elif self._is_manipulatable_object(shape_name):
                    self._set_friction(shape, self.object_friction, "object")
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not modify friction for some objects: {e}")
    
    def _is_gripper_component(self, name: str) -> bool:
        """Check if object is part of the gripper"""
        gripper_keywords = [
            'gripper', 'finger', 'panda_hand', 'panda_finger', 
            'panda_leftfinger', 'panda_rightfinger', 'hand'
        ]
        return any(keyword in name for keyword in gripper_keywords)
    
    def _is_table_surface(self, name: str) -> bool:
        """Check if object is a table or surface"""
        surface_keywords = [
            'table', 'surface', 'plane', 'floor', 'desk', 'counter'
        ]
        return any(keyword in name for keyword in surface_keywords)
    
    def _is_manipulatable_object(self, name: str) -> bool:
        """Check if object is a manipulatable item"""
        # Exclude gripper, table, and static environment objects
        exclude_keywords = [
            'gripper', 'finger', 'panda', 'hand', 'table', 'surface', 
            'plane', 'floor', 'wall', 'ceiling', 'camera', 'light',
            'joint', 'link', 'base', 'mount', 'stand'
        ]
        
        # If it contains exclude keywords, it's probably not manipulatable
        if any(keyword in name for keyword in exclude_keywords):
            return False
            
        # If it's a small, moveable object, it's probably manipulatable
        manipulatable_keywords = [
            'cube', 'box', 'ball', 'sphere', 'cylinder', 'object',
            'block', 'item', 'tool', 'bottle', 'cup', 'mug'
        ]
        
        return any(keyword in name for keyword in manipulatable_keywords)
    
    def _get_current_friction(self, shape: Shape) -> Optional[float]:
        """Get current friction coefficient of a shape"""
        try:
            # Try different physics engines
            try:
                return shape.get_bullet_friction()
            except:
                try:
                    return shape.get_ode_friction()
                except:
                    return None
        except Exception:
            return None
    
    def _set_friction(self, shape: Shape, friction: float, object_type: str):
        """Set friction coefficient for a shape"""
        try:
            shape_name = shape.get_name()
            original_friction = self._get_current_friction(shape)
            
            # Set friction for different physics engines
            try:
                shape.set_bullet_friction(friction)
            except:
                pass
                
            try:
                shape.set_ode_friction(friction)
            except:
                pass
            
            self.modified_objects.append({
                'name': shape_name,
                'type': object_type,
                'original_friction': original_friction,
                'new_friction': friction
            })
            
            if self.verbose:
                print(f"  {object_type}: {shape_name} - friction: {original_friction:.3f} -> {friction:.3f}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not set friction for {shape.get_name()}: {e}")
    
    def get_friction_info(self) -> Dict[str, Any]:
        """Get information about friction modifications"""
        return {
            'object_friction': self.object_friction,
            'gripper_friction': self.gripper_friction, 
            'table_friction': self.table_friction,
            'modified_objects': self.modified_objects
        }
    
    def print_friction_summary(self):
        """Print summary of friction modifications"""
        info = self.get_friction_info()
        
        print("\n=== Friction Modification Summary ===")
        print(f"Object friction: {info['object_friction']}")
        print(f"Gripper friction: {info['gripper_friction']}")
        print(f"Table friction: {info['table_friction']}")
        print(f"Modified {len(info['modified_objects'])} objects:")
        
        for obj in info['modified_objects']:
            print(f"  {obj['type']}: {obj['name']} ({obj['original_friction']:.3f} -> {obj['new_friction']:.3f})")
        print("======================================\n")


def create_high_friction_env(task_name: str,
                           object_friction: float = 2.0,
                           gripper_friction: float = 3.0,
                           table_friction: float = 1.5,
                           **kwargs) -> HighFrictionRlbenchEnv:
    """
    Convenience function to create a high-friction RLBench environment
    
    Args:
        task_name: Name of the RLBench task
        object_friction: Friction for manipulatable objects  
        gripper_friction: Friction for gripper components
        table_friction: Friction for table/surfaces
        **kwargs: Additional arguments for RlbenchEnv
    
    Returns:
        HighFrictionRlbenchEnv instance
    """
    env = HighFrictionRlbenchEnv(
        task_name=task_name,
        object_friction=object_friction,
        gripper_friction=gripper_friction,
        table_friction=table_friction,
        **kwargs
    )
    
    return env


if __name__ == "__main__":
    # Example usage
    print("Testing high-friction RLBench environment...")
    
    # Create environment with increased friction
    env = create_high_friction_env(
        task_name="pick_and_lift",
        object_friction=3.0,
        gripper_friction=4.0,
        table_friction=2.0,
        image_size=128,
        enable_depth=True,
        num_points=1024
    )
    
    # Reset and check friction modifications
    obs, info = env.reset()
    env.print_friction_summary()
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        
        if done:
            break
    
    env.close()
    print("Test completed!")