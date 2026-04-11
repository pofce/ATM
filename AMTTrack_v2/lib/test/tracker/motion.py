import numpy as np


class KalmanMotionModel:
    """
    Constant-velocity Kalman filter for bounding box tracking.

    State vector  x: [cx, cy, vx, vy, w, h]
    Observation   z: [cx, cy, w, h]   (derived from input [x1, y1, w, h])

    Transition model (F):
        cx  ← cx + vx
        cy  ← cy + vy
        vx  ← vx
        vy  ← vy
        w   ← w
        h   ← h

    Observation model (H):
        z = [cx, cy, w, h]  →  rows 0,1,4,5 of state
    """

    # ------------------------------------------------------------------ #
    # tuneable noise parameters                                            #
    # ------------------------------------------------------------------ #
    _Q_diag = np.array([1.0, 1.0,   # cx, cy  process noise
                        10.0, 10.0, # vx, vy  process noise (larger → faster adaptation)
                        1.0, 1.0],  # w,  h   process noise
                       dtype=np.float64)

    _R_diag = np.array([1.0, 1.0,   # cx, cy  measurement noise
                        10.0, 10.0], # w,  h   measurement noise
                       dtype=np.float64)

    def __init__(self):
        # state transition matrix
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = 1.0  # cx += vx
        self.F[1, 3] = 1.0  # cy += vy

        # observation matrix:  z = H @ x
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 4] = 1.0  # w
        self.H[3, 5] = 1.0  # h

        self.Q = np.diag(self._Q_diag)   # process noise covariance
        self.R = np.diag(self._R_diag)   # measurement noise covariance

        self.x = None   # state mean   (6,)
        self.P = None   # state covariance (6,6)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _xywh_to_cxcywh(box):
        """[x1, y1, w, h] → [cx, cy, w, h]"""
        x1, y1, w, h = box
        return np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=np.float64)

    @staticmethod
    def _cxcywh_to_xywh(cx, cy, w, h):
        """[cx, cy, w, h] → [x1, y1, w, h]"""
        return [cx - 0.5 * w, cy - 0.5 * h, w, h]

    def _is_initialised(self):
        return self.x is not None

    # ------------------------------------------------------------------ #
    # public API                                                           #
    # ------------------------------------------------------------------ #
    def update(self, state):
        """
        Ingest a new [x1, y1, w, h] observation and run a full
        predict → correct Kalman cycle.

        On the first call the filter is initialised from the observation
        (velocity set to zero, high initial uncertainty).

        Args:
            state: list or array [x1, y1, w, h] in image coordinates.
        """
        z = self._xywh_to_cxcywh(state)  # (4,)

        if not self._is_initialised():
            # initialise state from first observation
            self.x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3]],
                               dtype=np.float64)
            # high initial covariance for velocities
            self.P = np.diag([10.0, 10.0, 100.0, 100.0, 10.0, 10.0])
            return

        # --- predict step ---
        x_pred = self.F @ self.x                        # (6,)
        P_pred = self.F @ self.P @ self.F.T + self.Q    # (6,6)

        # --- update step ---
        y = z - self.H @ x_pred                         # innovation (4,)
        S = self.H @ P_pred @ self.H.T + self.R         # (4,4)
        K = P_pred @ self.H.T @ np.linalg.inv(S)        # (6,4) Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred

    def predict_only(self):
        """
        Advance the state by one time step without an observation.
        Intended for use during blind periods (low confidence / illumination
        change) where the network output is unreliable.

        Returns:
            list [x1, y1, w, h] — predicted bounding box in image coordinates.
            Returns last known state if the filter has not been initialised.
        """
        if not self._is_initialised():
            raise RuntimeError(
                "KalmanMotionModel.predict_only() called before any update().")

        # propagate mean and covariance forward (no measurement)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        cx, cy, _vx, _vy, w, h = self.x
        # clamp size to stay positive
        w = max(w, 1.0)
        h = max(h, 1.0)
        return self._cxcywh_to_xywh(cx, cy, w, h)

    def reset(self):
        """Clear filter state (call when target is re-acquired after long loss)."""
        self.x = None
        self.P = None
