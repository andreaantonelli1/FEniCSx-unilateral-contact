import os
import pickle
import shutil
import numpy as np

from dolfinx import fem
from dolfinx.io import XDMFFile, VTXWriter

class CheckpointManager:
    """Manages checkpointing for crash recovery"""
    
    def __init__(self, problem, output_dir):
        self.problem = problem
        self.output_dir = output_dir
        self.checkpoint_dir = f"{output_dir}/checkpoints"
        self.domain = self.problem.u.function_space.mesh
        self.comm = self.domain.comm
        self.rank = self.comm.rank
        
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.comm.barrier()

    def save_checkpoint(self, step, d_incr, d_val):
        """Save complete state to disk for crash recovery"""
        try:
            checkpoint_name = f"checkpoint_{step:04d}"
            checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_name}"

            # Create subdirectory for this checkpoint
            if self.rank == 0:
                os.makedirs(checkpoint_path, exist_ok=True)
            self.comm.barrier()
            
            # Save each function in a separate file with its own mesh
            functions_to_save = [
                ('u', self.problem.u)
            ]

            # Optionally save ray-tracing xi if present on problem
            if hasattr(self.problem, 'xi') and getattr(self.problem, 'xi', None) is not None:
                functions_to_save.append(('xi', self.problem.xi))
            
            for func_name, func in functions_to_save:
                # VTX output (supports any element order, including non-isoparametric)
                vtx_path = f"{checkpoint_path}/{func_name}_vtx"
                
                # Remove old VTX directory if it exists
                if self.rank == 0:
                    if os.path.exists(vtx_path):
                        shutil.rmtree(vtx_path)
                self.comm.barrier()
                
                # Set the function name
                func.name = func_name
                
                # Write using VTXWriter (supports higher-order elements)
                vtx = VTXWriter(self.comm, vtx_path, [func], engine="BP4")
                vtx.write(float(step))
                vtx.close()
                
                # --- XDMF OUTPUT (commented out - requires P1 interpolation for non-isoparametric elements) ---
                # file_path = f"{checkpoint_path}/{func_name}.xdmf"
                # 
                # # Remove old files if they exist
                # if self.rank == 0:
                #     if os.path.exists(file_path):
                #         os.remove(file_path)
                #     h5_file = file_path.replace('.xdmf', '.h5')
                #     if os.path.exists(h5_file):
                #         os.remove(h5_file)
                # self.comm.barrier()
                # 
                # # Write mesh and function together
                # with XDMFFile(self.comm, file_path, "w") as xdmf:
                #     # IMPORTANT: Write mesh BEFORE function
                #     xdmf.write_mesh(self.domain)
                #     # Set the function name
                #     func.name = func_name
                #     # Write function after mesh
                #     xdmf.write_function(func)
            
            # Save metadata
            if self.rank == 0:
                metadata = {
                    'step': step,
                    'current_d_incr': d_incr,
                    'd_val': d_val,
                }

                # Persist xi values in metadata for robust restart (fallback to VTX/XDMF)
                try:
                    if hasattr(self.problem, 'xi') and getattr(self.problem, 'xi', None) is not None:
                        metadata['xi'] = self.problem.xi.x.array.copy()
                except Exception:
                    pass
                
                with open(f"{checkpoint_path}/metadata.pkl", 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Update latest marker
                with open(f"{self.checkpoint_dir}/latest.txt", 'w') as f:
                    f.write(checkpoint_name)
                
                # Clean old checkpoints
                self._cleanup_old_checkpoints(keep_last=5)
            
            self.comm.barrier()
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"  -> Warning: Failed to save checkpoint: {e}")
                import traceback
                traceback.print_exc()
            return False

    def load_latest_checkpoint(self):
        """Load from disk after a crash"""
        try:
            checkpoint_path = None
            
            # Find latest checkpoint
            if self.rank == 0:
                latest_marker = f"{self.checkpoint_dir}/latest.txt"
                
                if os.path.exists(latest_marker):
                    with open(latest_marker, 'r') as f:
                        checkpoint_name = f.read().strip()
                    checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_name}"
                else:
                    # Fallback: find latest checkpoint directory
                    checkpoint_dirs = sorted([d for d in os.listdir(self.checkpoint_dir) 
                                            if d.startswith('checkpoint_') and 
                                            os.path.isdir(f"{self.checkpoint_dir}/{d}")])
                    if checkpoint_dirs:
                        checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_dirs[-1]}"
                
                # Verify checkpoint exists and has metadata
                if checkpoint_path and not os.path.exists(f"{checkpoint_path}/metadata.pkl"):
                    checkpoint_path = None
            
            # Broadcast checkpoint path
            checkpoint_path = self.comm.bcast(checkpoint_path, root=0)
            
            if checkpoint_path is None:
                return None
            
            # Load metadata
            metadata = None
            if self.rank == 0:
                with open(f"{checkpoint_path}/metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
            metadata = self.comm.bcast(metadata, root=0)

            # If metadata contains xi and problem has xi, restore numeric array into function
            try:
                if metadata is not None and 'xi' in metadata and hasattr(self.problem, 'xi'):
                    xi_arr = np.asarray(metadata['xi'])
                    tgt = self.problem.xi
                    if tgt.x.array.size == xi_arr.size:
                        tgt.x.array[:] = xi_arr
                        tgt.x.scatter_forward()
                        if self.rank == 0:
                            print(f"  -> Restored xi from metadata (length={xi_arr.size})")
                    else:
                        if self.rank == 0:
                            print(f"  -> Warning: xi size mismatch: checkpoint {xi_arr.size}, target {tgt.x.array.size}")
            except Exception:
                pass
            
            # Load functions from their individual files
            functions_to_load = {
                'u': self.problem.u,
            }
            
            for func_name, target_func in functions_to_load.items():
                file_path = f"{checkpoint_path}/{func_name}.xdmf"

                # If XDMF file is not present, check for VTX (BP4) directory
                if not os.path.exists(file_path):
                    vtx_path = f"{checkpoint_path}/{func_name}_vtx"
                    if os.path.isdir(vtx_path):
                        if self.rank == 0:
                            print(f"  -> Found VTX checkpoint at {vtx_path}; XDMF loader will skip it.")
                        # We don't have a VTX reader here; leave function as default or rely on other restart mechanisms
                        continue
                    else:
                        if self.rank == 0:
                            print(f"  -> Warning: Missing checkpoint file {file_path}")
                        continue
                
                # DOLFINx 0.8.0 doesn't have read_function method!y
                
                # Create a temporary function to read into
                temp_func = fem.Function(target_func.function_space)
                temp_func.name = func_name
                
                try:
                    # Try to read using the HDF5 file directly
                    h5_file_path = file_path.replace('.xdmf', '.h5')
                    
                    if os.path.exists(h5_file_path):
                        # Read the HDF5 file directly
                        import h5py
                        
                        with h5py.File(h5_file_path, 'r') as h5f:
                            # Look for the function data in the HDF5 file
                            # The data is stored under 'Function/{func_name}/0'
                            func_path = f'Function/{func_name}/0'
                            if func_path in h5f:
                                func_data = h5f[func_path][:]
                                expected_size = temp_func.x.array.size
                                
                                # Handle different array shapes
                                if func_data.size == expected_size:
                                    # Direct assignment if sizes match
                                    temp_func.x.array[:] = func_data.flatten()
                                    if self.rank == 0:
                                        print(f"  -> Successfully loaded {func_name} from checkpoint")
                                elif func_data.ndim == 2 and func_data.shape[1] == 1 and func_data.shape[0] == expected_size:
                                    # Handle (N,1) -> (N,) for scalar functions like damage
                                    temp_func.x.array[:] = func_data.flatten()
                                    if self.rank == 0:
                                        print(f"  -> Successfully loaded {func_name} from checkpoint (reshaped from {func_data.shape})")
                                elif func_data.ndim == 2 and func_data.shape[0] * func_data.shape[1] == expected_size:
                                    # Handle (N,4) -> (4*N,) for tensor functions like Cv
                                    temp_func.x.array[:] = func_data.flatten()
                                    if self.rank == 0:
                                        print(f"  -> Successfully loaded {func_name} from checkpoint (reshaped from {func_data.shape})")
                                elif func_data.ndim == 2 and func_data.shape[1] == 3 and func_data.shape[0] * 2 == expected_size:
                                    # Handle (N,3) -> (2*N,) for 2D displacement functions saved as 3D
                                    # Take only the first 2 components (x, y) and flatten
                                    temp_func.x.array[:] = func_data[:, :2].flatten()
                                    if self.rank == 0:
                                        print(f"  -> Successfully loaded {func_name} from checkpoint (reshaped from {func_data.shape} to 2D)")
                                else:
                                    if self.rank == 0:
                                        print(f"  -> Warning: Size mismatch for {func_name}: got {func_data.shape} (size {func_data.size}), expected size {expected_size}, using defaults")
                                    self._initialize_default_values(temp_func, func_name)
                            else:
                                if self.rank == 0:
                                    print(f"  -> Warning: No Function data found for {func_name}, using defaults")
                                self._initialize_default_values(temp_func, func_name)
                    else:
                        if self.rank == 0:
                            print(f"  -> Warning: HDF5 file not found for {func_name}, using defaults")
                        self._initialize_default_values(temp_func, func_name)
                
                except Exception as e:
                    if self.rank == 0:
                        print(f"  -> Error reading {func_name}: {e}")
                        # Provide more detailed error information for shape mismatches
                        if "could not broadcast input array" in str(e) or "shape" in str(e):
                            print(f"     This is likely a shape mismatch issue. Check function space definitions.")
                    self._initialize_default_values(temp_func, func_name)
                
                # Copy data to target function
                target_func.x.array[:] = temp_func.x.array[:]
                
                # Ensure ghost values are updated
                target_func.x.scatter_forward()
            
            if self.rank == 0:
                print(f"  -> Checkpoint loaded successfully!")
                print(f"     Step: {metadata['step']}")
                print(f"     Last d_incr: {metadata['current_d_incr']:.4f}")
                print(f"     Last d_val: {metadata['d_val']:.4f}")
            
            return metadata
            
        except Exception as e:
            if self.rank == 0:
                print(f"  -> Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()
            return None

    def _cleanup_old_checkpoints(self, keep_last=3):
        """Remove old checkpoint directories, keeping only the last N"""
        try:
            checkpoint_dirs = sorted([d for d in os.listdir(self.checkpoint_dir) 
                                    if d.startswith('checkpoint_') and
                                    os.path.isdir(f"{self.checkpoint_dir}/{d}")])
            
            if len(checkpoint_dirs) > keep_last:
                for old_dir in checkpoint_dirs[:-keep_last]:
                    dir_path = f"{self.checkpoint_dir}/{old_dir}"
                    shutil.rmtree(dir_path)
                    if self.rank == 0:
                        print(f"  -> Removed old checkpoint: {old_dir}")
        except Exception as e:
            if self.rank == 0:
                print(f"  -> Warning: Could not clean old checkpoints: {e}")
    
    def _initialize_default_values(self, func, func_name):
        """Initialize function with default values based on its type"""
        if func_name == 'Cv' or func_name == 'Cv_old':
            # Initialize Cv to identity tensor
            func.interpolate(self.problem._identity_tensor)
        else:
            # Initialize other functions to zero
            func.x.array[:] = 0.0
    
    def save_emergency_checkpoint(self, problem, step, reason="emergency"):
        """Quick emergency save when crash is imminent"""
        try:
            emergency_dir = f"{self.checkpoint_dir}/emergency_{step:03d}"
            
            if self.rank == 0:
                os.makedirs(emergency_dir, exist_ok=True)
            self.comm.barrier()
            
            # Save critical functions only using VTX (supports any element order)
            for func_name, func in [('u', problem.u)]:
                vtx_path = f"{emergency_dir}/{func_name}_vtx"
                func.name = func_name
                vtx = VTXWriter(self.comm, vtx_path, [func], engine="BP4")
                vtx.write(float(step))
                vtx.close()
                
                # --- XDMF OUTPUT (commented out - requires P1 interpolation for non-isoparametric elements) ---
                # file_path = f"{emergency_dir}/{func_name}.xdmf"
                # with XDMFFile(self.comm, file_path, "w") as xdmf:
                #     xdmf.write_mesh(self.domain)
                #     xdmf.write_function(func, 0.0)

            
            if self.rank == 0:
                with open(f"{emergency_dir}/info.txt", 'w') as f:
                    f.write(f"step={step}\nreason={reason}\n")
                print(f"  -> Emergency checkpoint saved: {reason}")
                
        except Exception as e:
            if self.rank == 0:
                print(f"  -> Failed to save emergency checkpoint: {e}")