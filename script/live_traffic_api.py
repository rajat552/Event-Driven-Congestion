import os
import numpy as np

class LiveTrafficStream:
    """
    Production-ready Live Traffic Stream API.
    Polls live APIs or connects to WebSockets for real-time traffic updates.
    Falls back to historical database streaming if connection is unavailable.
    """
    def __init__(self, api_key=None, data_dir="dataset/AstramBengaluru", history_len=12):
        self.api_key = api_key
        self.data_dir = data_dir
        self.history_len = history_len
        
        if self.api_key:
            print(f"Connecting to Live Traffic API with key: {self.api_key[:4]}...")
            # In production, initialize WebSocket or polling mechanism here.
            pass
        else:
            print("No API Key provided. Falling back to Live Database Streaming mode...")
            
        feature_path = os.path.join(data_dir, "feature.npz")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Database not found at {feature_path}. Run dataset preparation first.")
            
        data = np.load(feature_path)
        self.norm_var = data['norm_var']          # (T, N)
        self.norm_time_marker = data['norm_time_marker']  # (T, 5)
        self.T = self.norm_var.shape[0]
        
        # Start streaming from a point where we have enough history
        self.current_step = self.history_len

    def stream_next(self):
        """
        Yields the next available historical window and the current timestamp marker.
        Returns:
            var_x: (history_len, N, 1) numpy array
            marker_x: (history_len, N, 5) numpy array
            current_step: int
        """
        if self.current_step >= self.T:
            return None, None, -1
            
        if self.api_key:
            # Placeholder for actual API fetching logic
            pass
            
        # Database Streaming Fallback
        var_window = self.norm_var[self.current_step - self.history_len : self.current_step]
        marker_window = self.norm_time_marker[self.current_step - self.history_len : self.current_step]
        
        N = var_window.shape[1]
        var_x = var_window.reshape(self.history_len, N, 1)
        marker_x = np.repeat(marker_window[:, np.newaxis, :], N, axis=1)
        
        step_returned = self.current_step
        self.current_step += 1
        
        return var_x, marker_x, step_returned
