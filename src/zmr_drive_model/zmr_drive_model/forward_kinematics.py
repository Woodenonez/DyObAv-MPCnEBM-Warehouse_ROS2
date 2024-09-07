import numpy as np

def unicycle_model(state: np.ndarray, action: np.ndarray, ts: float, rk4:bool=True) -> np.ndarray:
    """Unicycle model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: linear speed and angular speed.
        rk4: If True, use Runge-Kutta 4 to refine the model.

    Returns:
        The next state.
    """
    def d_state_f(state, action):
        return ts * np.array([action[0]*np.cos(state[2]), action[0]*np.sin(state[2]), action[1]])
    if rk4:
        k1 = d_state_f(state, action)
        k2 = d_state_f(state + 0.5*k1, action)
        k3 = d_state_f(state + 0.5*k2, action)
        k4 = d_state_f(state + k3, action)
        d_state = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    else:
        d_state = d_state(state, action)

    return state + d_state